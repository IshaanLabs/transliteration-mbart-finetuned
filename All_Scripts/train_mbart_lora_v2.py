import argparse
import random
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets
from jiwer import cer
import sacrebleu

from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

random.seed(42)
torch.backends.cuda.matmul.allow_tf32 = True

LANGS = ["hi", "bn", "ta"]
LANG_CODE = {"hi": "hi_IN", "bn": "bn_IN", "ta": "ta_IN"}


def load_split(data_dir: Path, lang: str, split: str, max_samples: int | None):
    df = pd.read_csv(data_dir / lang / f"{split}.tsv", sep="\t")
    df["lang"] = lang

    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)

    return Dataset.from_pandas(df)


def build_datasets(data_dir: Path, max_train_samples: int, max_eval_samples: int):

    train_sets = []
    eval_sets = []

    for lang in LANGS:
        train_sets.append(load_split(data_dir, lang, "train", max_train_samples))
        eval_sets.append(load_split(data_dir, lang, "val", max_eval_samples))

    return concatenate_datasets(train_sets), concatenate_datasets(eval_sets)


def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    data_dir = Path(args.data_dir)

    train_ds, eval_ds = build_datasets(
        data_dir,
        args.max_train_samples,
        args.max_eval_samples
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        use_safetensors=True
    )

    # LoRA config
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )

    model = get_peft_model(base_model, lora_cfg)

    model.print_trainable_parameters()

    # add language token
    def add_lang(ex):

        code = LANG_CODE[ex["lang"]]

        ex["lang_code"] = code
        ex["input_text"] = ex["src"]

        return ex

    train_ds = train_ds.map(add_lang)
    eval_ds = eval_ds.map(add_lang)

    # tokenization
    def preprocess(batch):

        tokenizer.src_lang = "en_XX"

        lang_codes = batch["lang_code"]

        inputs = tokenizer(
            batch["input_text"],
            max_length=args.max_source_length,
            truncation=True,
            padding="max_length",
        )

       # tgt_texts = [
       #     f"{tokenizer.lang_code_to_token[lc]} {t}"
       #     for lc, t in zip(lang_codes, batch["tgt"])
       # ]


        tgt_texts = [
            f"{lc} {t}"
            for lc, t in zip(lang_codes, batch["tgt"])
        ]


        targets = tokenizer(
            text_target=tgt_texts,
            max_length=args.max_target_length,
            truncation=True,
            padding="max_length",
        )

        labels = targets["input_ids"]

        labels = [
            [(l if l != tokenizer.pad_token_id else -100) for l in seq]
            for seq in labels
        ]

        inputs["labels"] = labels

        inputs["forced_bos_token_id"] = [
            tokenizer.lang_code_to_id[lc] for lc in lang_codes
        ]

        return inputs

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    eval_tok = eval_ds.map(preprocess, batched=True, remove_columns=eval_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # evaluation metrics
    def compute_metrics(eval_preds):

        preds, labels = eval_preds

        preds = torch.tensor(preds)
        labels = torch.tensor(labels)

        if preds.dim() == 3:
            preds = preds.argmax(dim=-1)

        preds[preds < 0] = tokenizer.pad_token_id
        labels[labels < 0] = tokenizer.pad_token_id

        dec_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        dec_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        exact = sum(p == l for p, l in zip(dec_preds, dec_labels)) / len(dec_preds)

        cer_avg = sum(cer(l, p) for p, l in zip(dec_preds, dec_labels)) / len(dec_preds)

        bleu = sacrebleu.corpus_bleu(dec_preds, [dec_labels]).score

        return {
            "exact_match": exact,
            "cer": cer_avg,
            "bleu": bleu,
        }

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = Seq2SeqTrainingArguments(

        output_dir=str(args.output_dir),

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,

        gradient_accumulation_steps=args.grad_accum,

        learning_rate=args.lr,

        num_train_epochs=args.epochs,

        bf16=bf16,
        fp16=not bf16,

        eval_strategy="epoch",

        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,

        predict_with_generate=True,

        generation_max_length=args.max_target_length,

        lr_scheduler_type="cosine",

        warmup_steps=args.warmup_steps,

        gradient_checkpointing=True,

        max_grad_norm=1.0,

        optim="adafactor",

        logging_steps=50,

        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")

    trainer.train()

    print("Saving model...")

    trainer.save_model(str(args.output_dir))

    tokenizer.save_pretrained(str(args.output_dir))

    print("Training complete.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--model_name", default="facebook/mbart-large-50-many-to-many-mmt")

    parser.add_argument("--output_dir", default="outputs/mbart-lora")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=3)

    parser.add_argument("--lr", type=float, default=5.6e-05)

    parser.add_argument("--max_source_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=128)

    parser.add_argument("--max_train_samples", type=int, default=40000)
    parser.add_argument("--max_eval_samples", type=int, default=5000)

    parser.add_argument("--warmup_steps", type=int, default=500)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()

    main(args)
