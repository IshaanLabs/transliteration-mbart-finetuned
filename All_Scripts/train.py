import argparse
import random
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets
from jiwer import cer
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

def load_lang_split(data_dir: Path, lang: str, split: str, max_samples: int | None):
    path = data_dir / lang / f"{split}.tsv"
    print(f"Loading {path}")
    df = pd.read_csv(path, sep="\t")
    df["lang"] = lang
    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)
        print(f"  Sampled down to {len(df)} rows")
    return Dataset.from_pandas(df)

def build_datasets(data_dir: Path, max_train_samples: int, max_eval_samples: int):
    train_sets, eval_sets = [], []
    for lang in LANGS:
        train_sets.append(load_lang_split(data_dir, lang, "train", max_train_samples))
        eval_sets.append(load_lang_split(data_dir, lang, "val", max_eval_samples))
    train_ds = concatenate_datasets(train_sets)
    eval_ds = concatenate_datasets(eval_sets)
    print(f"Train total: {len(train_ds)}, Eval total: {len(eval_ds)}")
    return train_ds, eval_ds

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    data_dir = Path(args.data_dir)
    train_ds, eval_ds = build_datasets(data_dir, args.max_train_samples, args.max_eval_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, use_safetensors=True)

#    lora_cfg = LoraConfig(
#        r=24,              # higher capacity
#        lora_alpha=48,
#        target_modules=["q", "k", "v", "o"],
#        lora_dropout=0.05,
#        bias="none",
#        task_type="SEQ_2_SEQ_LM",
#    )

    lora_cfg = LoraConfig(
    r=24,
    lora_alpha=48,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],  # mBART names
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
    )



    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    def add_prefix(example):
        example["input_text"] = f"<{example['lang']}> {example['src']}"
        return example

    train_ds = train_ds.map(add_prefix)
    eval_ds = eval_ds.map(add_prefix)

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=args.max_source_length,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            text_target=batch["tgt"],
            max_length=args.max_target_length,
            truncation=True,
            padding="max_length",
        )
        label_ids = labels["input_ids"]
        label_ids = [
            [(lid if lid != tokenizer.pad_token_id else -100) for lid in seq]
            for seq in label_ids
        ]
        model_inputs["labels"] = label_ids
        return model_inputs

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    eval_tok = eval_ds.map(preprocess, batched=True, remove_columns=eval_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = torch.tensor(preds)
        labels = torch.tensor(labels)
        if preds.dim() == 3:
            preds = preds.argmax(dim=-1)
        preds[preds < 0] = tokenizer.pad_token_id
        labels[labels < 0] = tokenizer.pad_token_id
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        exact = sum(p == l for p, l in zip(decoded_preds, decoded_labels)) / len(decoded_preds)
        cer_scores = [cer(l, p) for p, l in zip(decoded_preds, decoded_labels)]
        cer_avg = sum(cer_scores) / len(cer_scores)
        return {"accuracy": exact, "cer": cer_avg}

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = not bf16
    eval_strategy = "epoch" if args.full_eval else "no"

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        bf16=bf16,
        fp16=fp16,
        eval_strategy=eval_strategy,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        optim="adafactor",
        load_best_model_at_end=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok if args.full_eval else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics if args.full_eval else None,
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    print("Training done.")

    if args.eval_subset > 0:
        print(f"Running eval on subset of {args.eval_subset} examples...")
        subset = eval_tok.select(range(min(args.eval_subset, len(eval_tok))))
        metrics = trainer.evaluate(eval_dataset=subset)
        print(metrics)

    print("Saving model...")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print("Saved to", args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--model_name", default="google/mt5-small")
    parser.add_argument("--output_dir", default="outputs/mt5-small-lora")
    parser.add_argument("--batch_size", type=int, default=16)      # L40S friendly
    parser.add_argument("--grad_accum", type=int, default=4)       # effective batch = 64
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_source_length", type=int, default=64)
    parser.add_argument("--max_target_length", type=int, default=64)
    parser.add_argument("--max_train_samples", type=int, default=80000)  # per language
    parser.add_argument("--max_eval_samples", type=int, default=5000)    # per language
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--full_eval", action="store_true", help="Run full eval each epoch (slower)")
    parser.add_argument("--eval_subset", type=int, default=5000, help="Subset eval after training (0 to skip)")
    parser.add_argument("--resume_from_checkpoint", default=None)
    args = parser.parse_args()
    main(args)

