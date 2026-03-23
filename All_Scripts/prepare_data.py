# # import argparse
# # import random
# # import unicodedata as ud
# # from pathlib import Path

# # import pandas as pd
# # from datasets import load_dataset

# # random.seed(42)

# # LANGS = {
# #     "hi": "hindi",
# #     "bn": "bengali",
# #     "ta": "tamil",
# # }

# # def ascii_ratio(text: str) -> float:
# #     if not text:
# #         return 0.0
# #     return sum(1 for ch in text if ch.isascii()) / len(text)

# # def pick_columns(example: dict):
# #     text_cols = [k for k, v in example.items() if isinstance(v, str)]
# #     if len(text_cols) < 2:
# #         raise ValueError("Need at least 2 text columns, found: %s" % text_cols)
# #     # Source = more ASCII (roman), Target = less ASCII (native)
# #     src = max(text_cols, key=lambda c: ascii_ratio(example[c]))
# #     tgt = [c for c in text_cols if c != src][0]
# #     print(f"  Picked source column: {src}")
# #     print(f"  Picked target column: {tgt}")
# #     return src, tgt

# # def clean_pair(src_text: str, tgt_text: str):
# #     # Lowercase roman input, trim spaces, normalize target to NFC
# #     src = " ".join(src_text.strip().lower().split())
# #     tgt = " ".join(tgt_text.strip().split())
# #     tgt = ud.normalize("NFC", tgt)
# #     return src, tgt

# # def maybe_make_val(train_list, val_list, val_frac=0.1):
# #     if val_list is not None:
# #         return train_list, val_list
# #     print(f"  No validation split found; creating {int(val_frac*100)}% from train")
# #     n = len(train_list)
# #     idx = list(range(n))
# #     random.shuffle(idx)
# #     cut = int(n * val_frac)
# #     val_idx = set(idx[:cut])
# #     new_train = [ex for i, ex in enumerate(train_list) if i not in val_idx]
# #     new_val = [ex for i, ex in enumerate(train_list) if i in val_idx]
# #     return new_train, new_val

# # def save_split(examples, src_key, tgt_key, path: Path, name: str):
# #     rows = []
# #     for ex in examples:
# #         src, tgt = clean_pair(ex[src_key], ex[tgt_key])
# #         if src and tgt:
# #             rows.append((src, tgt))
# #     df = pd.DataFrame(rows, columns=["src", "tgt"])
# #     path.parent.mkdir(parents=True, exist_ok=True)
# #     df.to_csv(path, sep="\t", index=False, encoding="utf-8")
# #     print(f"  Saved {name}: {len(df)} rows -> {path}")
# #     return len(df)

# # def main(out_dir: str):
# #     out_root = Path(out_dir)
# #     out_root.mkdir(parents=True, exist_ok=True)

# #     for code, name in LANGS.items():
# #         print(f"\n=== Processing {name} ({code}) ===")
# #         ds = load_dataset("ai4bharat/Aksharantar", code)

# #         # Detect column names using the first training example
# #         sample = ds["train"][0]
# #         src_key, tgt_key = pick_columns(sample)

# #         train_ds = ds["train"]
# #         val_ds = ds.get("validation") or ds.get("dev")
# #         test_ds = ds.get("test")

# #         train_ds, val_ds = maybe_make_val(train_ds, val_ds, val_frac=0.1)

# #         lang_dir = out_root / code
# #         save_split(train_ds, src_key, tgt_key, lang_dir / "train.tsv", "train")
# #         save_split(val_ds, src_key, tgt_key, lang_dir / "val.tsv", "val")
# #         if test_ds is not None:
# #             save_split(test_ds, src_key, tgt_key, lang_dir / "test.tsv", "test")
# #         else:
# #             print("  No test split provided; skipped.")

# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--out_dir", default="data/processed", help="Where to write TSV files")
# #     args = parser.parse_args()
# #     main(args.out_dir)




# import argparse
# import json
# import random
# import unicodedata as ud
# import zipfile
# from pathlib import Path
# import os
# import pandas as pd
# from huggingface_hub import hf_hub_download

# random.seed(42)

# LANGS = {
#     "hi": "hin.zip",
#     "bn": "ben.zip",
#     "ta": "tam.zip",
# }

# def ascii_ratio(text: str) -> float:
#     if not text:
#         return 0.0
#     return sum(1 for ch in text if ch.isascii()) / len(text)

# def pick_columns(example: dict):
#     text_cols = [k for k, v in example.items() if isinstance(v, str)]
#     if len(text_cols) < 2:
#         raise ValueError(f"Expected at least 2 text columns, got {text_cols}")
#     src = max(text_cols, key=lambda c: ascii_ratio(example[c]))  # roman
#     tgt = [c for c in text_cols if c != src][0]                  # native
#     print(f"    Picked source column: {src}")
#     print(f"    Picked target column: {tgt}")
#     return src, tgt

# def clean_pair(src_text: str, tgt_text: str):
#     src = " ".join(src_text.strip().lower().split())
#     tgt = " ".join(tgt_text.strip().split())
#     tgt = ud.normalize("NFC", tgt)
#     return src, tgt

# def read_split_from_zip(zip_path: Path, split_hint: str):
#     with zipfile.ZipFile(zip_path, "r") as zf:
#         names = zf.namelist()
#         split_file = None
#         for name in names:
#             low = name.lower()
#             if split_hint in low:
#                 split_file = name
#                 break
#         if split_file is None:
#             return []
#         print(f"    Reading {split_hint} from {split_file}")
#         with zf.open(split_file) as f:
#             data = json.load(f)
#         return data

# def maybe_make_val(train_list, val_list, val_frac=0.1):
#     if val_list:
#         return train_list, val_list
#     print(f"    No validation split; creating {int(val_frac*100)}% from train")
#     n = len(train_list)
#     idx = list(range(n))
#     random.shuffle(idx)
#     cut = int(n * val_frac)
#     val_idx = set(idx[:cut])
#     new_train = [ex for i, ex in enumerate(train_list) if i not in val_idx]
#     new_val = [ex for i, ex in enumerate(train_list) if i in val_idx]
#     return new_train, new_val

# def save_split(examples, src_key, tgt_key, path: Path, name: str):
#     rows = []
#     for ex in examples:
#         # ignore extra keys like "score"
#         src, tgt = clean_pair(ex.get(src_key, ""), ex.get(tgt_key, ""))
#         if src and tgt:
#             rows.append((src, tgt))
#     df = pd.DataFrame(rows, columns=["src", "tgt"])
#     path.parent.mkdir(parents=True, exist_ok=True)
#     df.to_csv(path, sep="\t", index=False, encoding="utf-8")
#     print(f"    Saved {name}: {len(df)} rows -> {path}")
#     return len(df)

# def process_language(code: str, zip_name: str, out_root: Path):
#     print(f"\n=== Processing {code} ===")
#     # zip_path = Path(hf_hub_download("ai4bharat/Aksharantar", filename=zip_name))
#     zip_path = Path(
#     hf_hub_download(
#         "ai4bharat/Aksharantar",
#         filename=zip_name,
#         repo_type="dataset",
#         token=os.getenv("HF_TOKEN"),
#     )
# )

#     print(f"  Downloaded: {zip_path}")

#     train_raw = read_split_from_zip(zip_path, "train")
#     val_raw = read_split_from_zip(zip_path, "val")
#     test_raw = read_split_from_zip(zip_path, "test")

#     if not train_raw:
#         raise RuntimeError(f"No train split found in {zip_name}")

#     sample = train_raw[0]
#     src_key, tgt_key = pick_columns(sample)

#     train_raw, val_raw = maybe_make_val(train_raw, val_raw, val_frac=0.1)

#     lang_dir = out_root / code
#     save_split(train_raw, src_key, tgt_key, lang_dir / "train.tsv", "train")
#     save_split(val_raw, src_key, tgt_key, lang_dir / "val.tsv", "val")
#     if test_raw:
#         save_split(test_raw, src_key, tgt_key, lang_dir / "test.tsv", "test")
#     else:
#         print("    No test split found; skipped.")

# def main(out_dir: str):
#     out_root = Path(out_dir)
#     out_root.mkdir(parents=True, exist_ok=True)
#     for code, zip_name in LANGS.items():
#         process_language(code, zip_name, out_root)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--out_dir", default="data/processed", help="Where to write TSV files")
#     args = parser.parse_args()
#     main(args.out_dir)




# import argparse
# import json
# import os
# import random
# import unicodedata as ud
# import zipfile
# from pathlib import Path

# import pandas as pd
# from huggingface_hub import hf_hub_download

# random.seed(42)

# LANGS = {
#     "hi": "hin.zip",
#     "bn": "ben.zip",
#     "ta": "tam.zip",
# }

# def ascii_ratio(text: str) -> float:
#     if not text:
#         return 0.0
#     return sum(1 for ch in text if ch.isascii()) / len(text)

# def pick_columns(example: dict):
#     text_cols = [k for k, v in example.items() if isinstance(v, str)]
#     if len(text_cols) < 2:
#         raise ValueError(f"Expected at least 2 text columns, got {text_cols}")
#     src = max(text_cols, key=lambda c: ascii_ratio(example[c]))  # roman
#     tgt = [c for c in text_cols if c != src][0]                  # native
#     print(f"    Picked source column: {src}")
#     print(f"    Picked target column: {tgt}")
#     return src, tgt

# def clean_pair(src_text: str, tgt_text: str):
#     src = " ".join(src_text.strip().lower().split())
#     tgt = " ".join(tgt_text.strip().split())
#     tgt = ud.normalize("NFC", tgt)
#     return src, tgt

# def read_split_from_zip(zip_path: Path, split_hint: str):
#     with zipfile.ZipFile(zip_path, "r") as zf:
#         names = zf.namelist()
#         split_file = None
#         for name in names:
#             low = name.lower()
#             if split_hint in low:
#                 split_file = name
#                 break
#         if split_file is None:
#             return []
#         print(f"    Reading {split_hint} from {split_file}")
#         with zf.open(split_file) as f:
#             raw_bytes = f.read()
#         text = raw_bytes.decode("utf-8")
#         try:
#             return json.loads(text)
#         except json.JSONDecodeError:
#             rows = []
#             for line in text.splitlines():
#                 line = line.strip()
#                 if not line:
#                     continue
#                 try:
#                     rows.append(json.loads(line))
#                 except json.JSONDecodeError:
#                     continue
#             if not rows:
#                 raise
#             return rows

# def maybe_make_val(train_list, val_list, val_frac=0.1):
#     if val_list:
#         return train_list, val_list
#     print(f"    No validation split; creating {int(val_frac*100)}% from train")
#     n = len(train_list)
#     idx = list(range(n))
#     random.shuffle(idx)
#     cut = int(n * val_frac)
#     val_idx = set(idx[:cut])
#     new_train = [ex for i, ex in enumerate(train_list) if i not in val_idx]
#     new_val = [ex for i, ex in enumerate(train_list) if i in val_idx]
#     return new_train, new_val

# def save_split(examples, src_key, tgt_key, path: Path, name: str):
#     rows = []
#     for ex in examples:
#         src, tgt = clean_pair(ex.get(src_key, ""), ex.get(tgt_key, ""))
#         if src and tgt:
#             rows.append((src, tgt))
#     df = pd.DataFrame(rows, columns=["src", "tgt"])
#     path.parent.mkdir(parents=True, exist_ok=True)
#     df.to_csv(path, sep="\t", index=False, encoding="utf-8")
#     print(f"    Saved {name}: {len(df)} rows -> {path}")
#     return len(df)

# def process_language(code: str, zip_name: str, out_root: Path):
#     print(f"\n=== Processing {code} ===")
#     zip_path = Path(
#         hf_hub_download(
#             "ai4bharat/Aksharantar",
#             filename=zip_name,
#             repo_type="dataset",
#             token=os.getenv("HF_TOKEN"),
#         )
#     )
#     print(f"  Downloaded: {zip_path}")

#     train_raw = read_split_from_zip(zip_path, "train")
#     val_raw = read_split_from_zip(zip_path, "val")
#     test_raw = read_split_from_zip(zip_path, "test")

#     if not train_raw:
#         raise RuntimeError(f"No train split found in {zip_name}")

#     sample = train_raw[0]
#     src_key, tgt_key = pick_columns(sample)

#     train_raw, val_raw = maybe_make_val(train_raw, val_raw, val_frac=0.1)

#     lang_dir = out_root / code
#     save_split(train_raw, src_key, tgt_key, lang_dir / "train.tsv", "train")
#     save_split(val_raw, src_key, tgt_key, lang_dir / "val.tsv", "val")
#     if test_raw:
#         save_split(test_raw, src_key, tgt_key, lang_dir / "test.tsv", "test")
#     else:
#         print("    No test split found; skipped.")

# def main(out_dir: str):
#     out_root = Path(out_dir)
#     out_root.mkdir(parents=True, exist_ok=True)
#     for code, zip_name in LANGS.items():
#         process_language(code, zip_name, out_root)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--out_dir", default="data/processed", help="Where to write TSV files")
#     args = parser.parse_args()
#     main(args.out_dir)



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


