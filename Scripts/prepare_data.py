import argparse
import json
import os
import random
import unicodedata as ud
import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

random.seed(42)

LANGS = {
    "hi": "hin.zip",
    "bn": "ben.zip",
    "ta": "tam.zip",
}

def find_column(keys, target):
    target = target.lower()
    for k in keys:
        if k.lower() == target:
            return k
    return None

def clean_pair(src_text: str, tgt_text: str):
    src = " ".join(src_text.strip().lower().split())
    tgt = " ".join(tgt_text.strip().split())
    tgt = ud.normalize("NFC", tgt)
    return src, tgt

def read_split_from_zip(zip_path: Path, split_hint: str):
    with zipfile.ZipFile(zip_path, "r") as zf:
        split_file = None
        for name in zf.namelist():
            if split_hint in name.lower():
                split_file = name
                break
        if split_file is None:
            return []
        print(f"    Reading {split_hint} from {split_file}")
        with zf.open(split_file) as f:
            text = f.read().decode("utf-8")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            rows = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return rows

def maybe_make_val(train_list, val_list, val_frac=0.1):
    if val_list:
        return train_list, val_list
    print(f"    No validation split; creating {int(val_frac*100)}% from train")
    idx = list(range(len(train_list)))
    random.shuffle(idx)
    cut = int(len(train_list) * val_frac)
    val_idx = set(idx[:cut])
    new_train = [ex for i, ex in enumerate(train_list) if i not in val_idx]
    new_val = [ex for i, ex in enumerate(train_list) if i in val_idx]
    return new_train, new_val

def save_split(examples, src_key, tgt_key, path: Path, name: str):
    rows = []
    for ex in examples:
        src, tgt = clean_pair(ex.get(src_key, ""), ex.get(tgt_key, ""))
        if src and tgt:
            rows.append((src, tgt))
    df = pd.DataFrame(rows, columns=["src", "tgt"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, encoding="utf-8")
    print(f"    Saved {name}: {len(df)} rows -> {path}")

def process_language(code: str, zip_name: str, out_root: Path):
    print(f"\n=== Processing {code} ===")
    zip_path = Path(
        hf_hub_download(
            "ai4bharat/Aksharantar",
            filename=zip_name,
            repo_type="dataset",
            token=os.getenv("HF_TOKEN"),
        )
    )
    print(f"  Downloaded: {zip_path}")

    train_raw = read_split_from_zip(zip_path, "train")
    val_raw = read_split_from_zip(zip_path, "val")
    test_raw = read_split_from_zip(zip_path, "test")

    if not train_raw:
        raise RuntimeError(f"No train split found in {zip_name}")

    sample = train_raw[0]
    keys = list(sample.keys())
    src_key = find_column(keys, "english word")
    tgt_key = find_column(keys, "native word")
    if not src_key or not tgt_key:
        raise ValueError(f"Could not find english/native columns in keys: {keys}")
    print(f"    Using source: {src_key}")
    print(f"    Using target: {tgt_key}")

    train_raw, val_raw = maybe_make_val(train_raw, val_raw, val_frac=0.1)

    lang_dir = out_root / code
    save_split(train_raw, src_key, tgt_key, lang_dir / "train.tsv", "train")
    save_split(val_raw, src_key, tgt_key, lang_dir / "val.tsv", "val")
    if test_raw:
        save_split(test_raw, src_key, tgt_key, lang_dir / "test.tsv", "test")
    else:
        print("    No test split; skipped.")

def main(out_dir: str):
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    for code, zip_name in LANGS.items():
        process_language(code, zip_name, out_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/processed", help="Where to write TSV files")
    args = parser.parse_args()
    main(args.out_dir)


