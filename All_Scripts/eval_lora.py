import argparse
from pathlib import Path
import random
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets
from jiwer import cer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import PeftModel

random.seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
LANGS = ["hi", "bn", "ta"]

def load_split(data_dir: Path, lang: str, split: str, max_samples: int | None):
    df = pd.read_csv(data_dir / lang / f"{split}.tsv", sep="\t")
    df["lang"] = lang
    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)
    return Dataset.from_pandas(df)

def build_eval(data_dir: Path, max_eval_samples: int):
    return concatenate_datasets([load_split(data_dir, lang, "val", max_eval_samples) for lang in LANGS])

def main(args):
    data_dir = Path(args.data_dir)
    eval_ds = build_eval(data_dir, args.max_eval_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    base = AutoModelForSeq2SeqLM.from_pretrained(args.base_model, use_safetensors=True)
    model = PeftModel.from_pretrained(base, args.model_dir)
    model.eval()

    def add_prefix(ex):
        ex["input_text"] = f"<{ex['lang']}> {ex['src']}"
        return ex

    eval_ds = eval_ds.map(add_prefix)

    def preprocess(batch):
        mi = tokenizer(batch["input_text"], max_length=args.max_source_length, truncation=True, padding="max_length")
        labels = tokenizer(text_target=batch["tgt"], max_length=args.max_target_length, truncation=True, padding="max_length")
        li = labels["input_ids"]
        mi["labels"] = [[(lid if lid != tokenizer.pad_token_id else -100) for lid in seq] for seq in li]
        return mi

    eval_tok = eval_ds.map(preprocess, batched=True, remove_columns=eval_ds.column_names)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

    def compute_metrics(preds_labels):
        preds, labels = preds_labels
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
        return {"accuracy": exact, "cer": cer_avg}

    args_train = Seq2SeqTrainingArguments(
        output_dir="tmp-eval",
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        fp16=True,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args_train,
        eval_dataset=eval_tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()
    print(metrics)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
#    p.add_argument("--model_dir", default="outputs/mt5-small-lora-l40s")
#    p.add_argument("--base_model", default="google/mt5-small")
    p.add_argument("--model_dir", default="outputs/mbart-lora")  # Changed from mt5
    p.add_argument("--base_model", default="facebook/mbart-large-50")  # Changed from google/mt5-small
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_source_length", type=int, default=64)
    p.add_argument("--max_target_length", type=int, default=64)
    p.add_argument("--max_eval_samples", type=int, default=5000)
    args = p.parse_args()
    main(args)

