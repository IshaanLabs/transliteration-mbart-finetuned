import argparse
import json
import random
import unicodedata as ud
import zipfile
from pathlib import Path
import requests
import tempfile

import pandas as pd

random.seed(42)

LANGS = {
    "hi": ("hin", "hindi"),
    "bn": ("ben", "bengali"), 
    "ta": ("tam", "tamil"),
}

BASE_URL = "https://huggingface.co/datasets/ai4bharat/Aksharantar/resolve/main/"

def ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if ch.isascii()) / len(text)

def pick_columns(example: dict):
    text_cols = [k for k, v in example.items() if isinstance(v, str)]
    if len(text_cols) < 2:
        raise ValueError("Need at least 2 text columns, found: %s" % text_cols)
    src = max(text_cols, key=lambda c: ascii_ratio(example[c]))
    tgt = [c for c in text_cols if c != src][0]
    print(f"  Picked source column: {src}")
    print(f"  Picked target column: {tgt}")
    return src, tgt

def clean_pair(src_text: str, tgt_text: str):
    src = " ".join(src_text.strip().lower().split())
    tgt = " ".join(tgt_text.strip().split())
    tgt = ud.normalize("NFC", tgt)
    return src, tgt

def load_language_data(lang_code):
    """Download and load data for a specific language"""
    zip_url = f"{BASE_URL}{lang_code}.zip"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download zip file
        print(f"  Downloading {zip_url}")
        response = requests.get(zip_url)
        response.raise_for_status()
        
        zip_path = Path(temp_dir) / f"{lang_code}.zip"
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        # Extract and load JSON files
        data = {"train": [], "test": []}
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.filename.endswith('.json'):
                    with zip_ref.open(file_info) as json_file:
                        content = json_file.read().decode('utf-8')
                        json_data = json.loads(content)
                        
                        if 'train' in file_info.filename:
                            data["train"].extend(json_data)
                        elif 'test' in file_info.filename:
                            data["test"].extend(json_data)
        
        return data

def maybe_make_val(train_list, val_frac=0.1):
    print(f"  Creating {int(val_frac*100)}% validation split from train")
    n = len(train_list)
    idx = list(range(n))
    random.shuffle(idx)
    cut = int(n * val_frac)
    val_idx = set(idx[:cut])
    new_train = [ex for i, ex in enumerate(train_list) if i not in val_idx]
    new_val = [ex for i, ex in enumerate(train_list) if i in val_idx]
    return new_train, new_val

def save_split(examples, src_key, tgt_key, path: Path, name: str):
    rows = []
    for ex in examples:
        src, tgt = clean_pair(ex[src_key], ex[tgt_key])
        if src and tgt:
            rows.append((src, tgt))
    df = pd.DataFrame(rows, columns=["src", "tgt"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, encoding="utf-8")
    print(f"  Saved {name}: {len(df)} rows -> {path}")
    return len(df)

def main(out_dir: str):
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for code, (lang_code, name) in LANGS.items():
        print(f"\n=== Processing {name} ({code}) ===")
        
        try:
            data = load_language_data(lang_code)
            
            if not data["train"]:
                print(f"  No training data found for {name}")
                continue
                
            # Use first example to detect columns
            src_key, tgt_key = pick_columns(data["train"][0])
            
            # Create train/val split
            train_list, val_list = maybe_make_val(data["train"], val_frac=0.1)
            
            # Save splits
            lang_dir = out_root / code
            save_split(train_list, src_key, tgt_key, lang_dir / "train.tsv", "train")
            save_split(val_list, src_key, tgt_key, lang_dir / "val.tsv", "val")
            
            if data["test"]:
                save_split(data["test"], src_key, tgt_key, lang_dir / "test.tsv", "test")
            else:
                print("  No test split found; skipped.")
                
        except Exception as e:
            print(f"  Error processing {name}: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/processed", help="Where to write TSV files")
    args = parser.parse_args()
    main(args.out_dir)