# # eval_mbart_lora.py
# import argparse
# from pathlib import Path
# import pandas as pd
# import torch
# from datasets import Dataset, concatenate_datasets
# from jiwer import cer
# import sacrebleu
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from peft import PeftModel

# device = "cuda" if torch.cuda.is_available() else "cpu"
# #model.to(device)



# LANGS = ["hi", "bn", "ta"]
# LANG_CODE = {"hi": "hi_IN", "bn": "bn_IN", "ta": "ta_IN"}

# def load_split(data_dir: Path, lang: str, split: str, max_samples: int | None):
#     df = pd.read_csv(data_dir / lang / f"{split}.tsv", sep="\t")
#     df["lang"] = lang
#     if max_samples and len(df) > max_samples:
#         df = df.sample(max_samples, random_state=42)
#     return Dataset.from_pandas(df)

# def build_eval_dataset(data_dir: Path, max_eval_samples: int):
#     eval_sets = []
#     for lang in LANGS:
#         eval_sets.append(load_split(data_dir, lang, "val", max_eval_samples))
#     return concatenate_datasets(eval_sets)

# def main(args):
#     print(f"Loading model from {args.model_dir}")
    
#     # Load tokenizer and model
#     tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
#     base_model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model, use_safetensors=True)
# #    model = PeftModel.from_pretrained(base_model, args.model_dir)
# #    model.eval()

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     model = PeftModel.from_pretrained(base_model, args.model_dir)
#     model.to(device)
#     model.eval()

#     #device = "cuda" if torch.cuda.is_available() else "cpu"
#     #model.to(device)


#     # Load evaluation data
#     data_dir = Path(args.data_dir)
#     eval_ds = build_eval_dataset(data_dir, args.max_eval_samples)
    
#     # Add language processing
#     def add_lang(ex):
#         code = LANG_CODE[ex["lang"]]
#         ex["lang_code"] = code
#         ex["input_text"] = ex["src"]
#         return ex
    
#     eval_ds = eval_ds.map(add_lang)
    
#     # Generate predictions
#     predictions = []
#     references = []
    
#     print(f"Evaluating {len(eval_ds)} samples...")
    
#     for i, example in enumerate(eval_ds):
#         if i % 500 == 0:
#             print(f"Processing {i}/{len(eval_ds)}")
            
#         # Prepare input
#         tokenizer.src_lang = "en_XX"
#         #inputs = tokenizer(
#         #    example["input_text"], 
#         #    return_tensors="pt", 
#         #    max_length=args.max_source_length,
#         #    truncation=True
#         #)
#         inputs = tokenizer(
#             example["input_text"],
#             return_tensors="pt",
#             max_length=args.max_source_length,
#             truncation=True
#         ).to(device)


#         # Generate with forced language token
#         lang_token_id = tokenizer.lang_code_to_id[example["lang_code"]]
        
#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 forced_bos_token_id=lang_token_id,
#                 max_length=args.max_target_length,
#                 num_beams=4,
#                 early_stopping=True,
#                 do_sample=False
#             )
        
#         # Decode prediction
#         pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         ref = example["tgt"]
        
#         predictions.append(pred)
#         references.append(ref)
    
#     # Compute metrics
#     print("\nComputing metrics...")
    
#     # Exact match accuracy
#     exact_matches = sum(p == r for p, r in zip(predictions, references))
#     exact_match_acc = exact_matches / len(predictions)
    
#     # Character Error Rate
#     cer_scores = [cer(r, p) for p, r in zip(predictions, references)]
#     avg_cer = sum(cer_scores) / len(cer_scores)
    
#     # BLEU score
#     bleu_score = sacrebleu.corpus_bleu(predictions, [references]).score
    
#     # Per-language breakdown
#     lang_metrics = {}
#     for lang in LANGS:
#         lang_preds = [p for p, ex in zip(predictions, eval_ds) if ex["lang"] == lang]
#         lang_refs = [r for r, ex in zip(references, eval_ds) if ex["lang"] == lang]
        
#         if lang_preds:
#             lang_exact = sum(p == r for p, r in zip(lang_preds, lang_refs)) / len(lang_preds)
#             lang_cer = sum(cer(r, p) for p, r in zip(lang_preds, lang_refs)) / len(lang_preds)
#             lang_bleu = sacrebleu.corpus_bleu(lang_preds, [lang_refs]).score
            
#             lang_metrics[lang] = {
#                 "exact_match": lang_exact,
#                 "cer": lang_cer,
#                 "bleu": lang_bleu,
#                 "samples": len(lang_preds)
#             }
    
#     # Print results
#     print("\n" + "="*50)
#     print("EVALUATION RESULTS")
#     print("="*50)
#     print(f"Total samples: {len(predictions)}")
#     print(f"Exact Match Accuracy: {exact_match_acc:.4f} ({exact_matches}/{len(predictions)})")
#     print(f"Character Error Rate: {avg_cer:.4f}")
#     print(f"BLEU Score: {bleu_score:.2f}")
    
#     print("\nPer-language results:")
#     for lang, metrics in lang_metrics.items():
#         print(f"\n{lang.upper()} ({metrics['samples']} samples):")
#         print(f"  Exact Match: {metrics['exact_match']:.4f}")
#         print(f"  CER: {metrics['cer']:.4f}")
#         print(f"  BLEU: {metrics['bleu']:.2f}")
    
#     # Show some examples
#     print("\nSample predictions:")
#     for i in range(min(10, len(predictions))):
#         lang = eval_ds[i]["lang"]
#         src = eval_ds[i]["src"]
#         print(f"\n{lang}: {src}")
#         print(f"Pred: {predictions[i]}")
#         print(f"True: {references[i]}")
#         print(f"Match: {'✓' if predictions[i] == references[i] else '✗'}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_dir", default="outputs/mbart-lora-run-4")
#     parser.add_argument("--base_model", default="facebook/mbart-large-50-many-to-many-mmt")
#     parser.add_argument("--data_dir", default="data/processed")
#     parser.add_argument("--max_eval_samples", type=int, default=5000)
#     parser.add_argument("--max_source_length", type=int, default=128)
#     parser.add_argument("--max_target_length", type=int, default=128)
#     args = parser.parse_args()
#     main(args)








#########################


# eval_mbart_lora.py
import argparse
from pathlib import Path
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets
from jiwer import cer
import sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"

LANGS = ["hi", "bn", "ta"]
LANG_CODE = {"hi": "hi_IN", "bn": "bn_IN", "ta": "ta_IN"}

def normalize(text):
    return text.strip().replace(" ", "")

def load_split(data_dir: Path, lang: str, split: str, max_samples: int | None):
    df = pd.read_csv(data_dir / lang / f"{split}.tsv", sep="\t")
    df["lang"] = lang
    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)
    return Dataset.from_pandas(df)

def build_eval_dataset(data_dir: Path, max_eval_samples: int):
    eval_sets = []
    for lang in LANGS:
        eval_sets.append(load_split(data_dir, lang, "val", max_eval_samples))
    return concatenate_datasets(eval_sets)

def main(args):
    print(f"Loading model from {args.model_dir}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model, use_safetensors=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PeftModel.from_pretrained(base_model, args.model_dir)
    model.to(device)
    model.eval()

    # Load evaluation data
    data_dir = Path(args.data_dir)
    eval_ds = build_eval_dataset(data_dir, args.max_eval_samples)
    
#    # Add language processing
#    def add_lang(ex):
#        code = LANG_CODE[ex["lang"]]
#        ex["lang_code"] = code
#        ex["input_text"] = ex["src"]
#        return ex

    def add_lang(ex):
        code = LANG_CODE[ex["lang"]]
        ex["lang_code"] = code
        ex["input_text"] = " ".join(list(ex["src"]))  # Match training format
        return ex
    
    eval_ds = eval_ds.map(add_lang)
    
    # Generate predictions
    predictions = []
    references = []
    
    print(f"Evaluating {len(eval_ds)} samples...")
    
    for i, example in enumerate(eval_ds):
        if i % 500 == 0:
            print(f"Processing {i}/{len(eval_ds)}")
            
        # Prepare input
        tokenizer.src_lang = "en_XX"
        inputs = tokenizer(
            example["input_text"],
            return_tensors="pt",
            max_length=args.max_source_length,
            truncation=True
        ).to(device)

        # Generate with forced language token
        lang_token_id = tokenizer.lang_code_to_id[example["lang_code"]]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=lang_token_id,
                max_length=args.max_target_length,
                num_beams=4,
                early_stopping=True,
                do_sample=False
            )
        
        # Decode prediction
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        ref = example["tgt"]
        
        predictions.append(pred)
        references.append(ref)
    
    # Normalize spaces
    norm_preds = [normalize(p) for p in predictions]
    norm_refs = [normalize(r) for r in references]

    # Compute metrics
    print("\nComputing metrics...")
    
    # Exact match accuracy
    exact_matches = sum(p == r for p, r in zip(norm_preds, norm_refs))
    exact_match_acc = exact_matches / len(norm_preds)
    
    # Character Error Rate
    cer_scores = [cer(r, p) for p, r in zip(norm_preds, norm_refs)]
    avg_cer = sum(cer_scores) / len(cer_scores)
    
    # BLEU score
    # bleu_score = sacrebleu.corpus_bleu(norm_preds, [norm_refs]).score

    # Replace this:
    bleu_score = sacrebleu.corpus_bleu(norm_preds, [norm_refs]).score

    # With character-level BLEU:
    char_preds = [" ".join(list(p)) for p in norm_preds]
    char_refs = [" ".join(list(r)) for r in norm_refs]
    bleu_score = sacrebleu.corpus_bleu(char_preds, [char_refs]).score

    
    # Per-language breakdown
    lang_metrics = {}
    for lang in LANGS:
        lang_preds = [p for p, ex in zip(norm_preds, eval_ds) if ex["lang"] == lang]
        lang_refs = [r for r, ex in zip(norm_refs, eval_ds) if ex["lang"] == lang]
        
        if lang_preds:
            lang_exact = sum(p == r for p, r in zip(lang_preds, lang_refs)) / len(lang_preds)
            lang_cer = sum(cer(r, p) for p, r in zip(lang_preds, lang_refs)) / len(lang_preds)
            lang_bleu = sacrebleu.corpus_bleu(lang_preds, [lang_refs]).score
            
            lang_metrics[lang] = {
                "exact_match": lang_exact,
                "cer": lang_cer,
                "bleu": lang_bleu,
                "samples": len(lang_preds)
            }
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total samples: {len(predictions)}")
    print(f"Exact Match Accuracy: {exact_match_acc:.4f} ({exact_matches}/{len(predictions)})")
    print(f"Character Error Rate: {avg_cer:.4f}")
    print(f"BLEU Score: {bleu_score:.2f}")
    
    print("\nPer-language results:")
    for lang, metrics in lang_metrics.items():
        print(f"\n{lang.upper()} ({metrics['samples']} samples):")
        print(f"  Exact Match: {metrics['exact_match']:.4f}")
        print(f"  CER: {metrics['cer']:.4f}")
        print(f"  BLEU: {metrics['bleu']:.2f}")
    
    # Show some examples
    print("\nSample predictions:")
    for i in range(min(10, len(predictions))):
        lang = eval_ds[i]["lang"]
        src = eval_ds[i]["src"]
        print(f"\n{lang}: {src}")
        print(f"Pred: {predictions[i]} -> {norm_preds[i]}")
        print(f"True: {references[i]} -> {norm_refs[i]}")
        print(f"Match: {'✓' if norm_preds[i] == norm_refs[i] else '✗'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="outputs/mbart-lora-run-4")
    parser.add_argument("--base_model", default="facebook/mbart-large-50-many-to-many-mmt")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--max_eval_samples", type=int, default=5000)
    parser.add_argument("--max_source_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=128)
    args = parser.parse_args()
    main(args)
