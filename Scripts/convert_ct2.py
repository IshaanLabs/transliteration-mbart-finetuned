# Scripts/convert_ct2.py
import argparse
import time
from pathlib import Path

import ctranslate2
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import MBart50TokenizerFast

LANG_CODE = {"hi": "hi_IN", "bn": "bn_IN", "ta": "ta_IN"}

#def merge_and_save(base_model_name, lora_dir, merged_dir):
#    print("Loading base model...")
#    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, use_safetensors=True)
#    print(f"Loading LoRA adapter from {lora_dir}...")
#    model = PeftModel.from_pretrained(base_model, lora_dir)
#    print("Merging LoRA weights...")
#    model = model.merge_and_unload()
#    print(f"Saving merged model to {merged_dir}...")
#    model.save_pretrained(merged_dir)
#    tokenizer = AutoTokenizer.from_pretrained(lora_dir)
#    tokenizer.save_pretrained(merged_dir)
#    print("Merged model saved.")

def merge_and_save(base_model_name, lora_dir, merged_dir):
    from transformers import MBart50TokenizerFast
    print("Loading base model...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, use_safetensors=True)
    print(f"Loading LoRA adapter from {lora_dir}...")
    model = PeftModel.from_pretrained(base_model, lora_dir)
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    print(f"Saving merged model to {merged_dir}...")
    model.save_pretrained(merged_dir)
    # Save the mBART-specific tokenizer (not AutoTokenizer)
    tokenizer = MBart50TokenizerFast.from_pretrained(base_model_name)
    tokenizer.save_pretrained(merged_dir)
    print("Merged model saved.")



def convert_to_ct2(merged_dir, ct2_dir, quantization="int8"):
    print(f"Converting to CTranslate2 (quantization={quantization})...")
    converter = ctranslate2.converters.TransformersConverter(merged_dir)
    converter.convert(ct2_dir, quantization=quantization, force=True)
    print(f"CTranslate2 model saved to {ct2_dir}")

def get_dir_size_mb(path):
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return total / (1024 * 1024)

def benchmark_pytorch(merged_dir, tokenizer, samples, device="cuda"):
    """Benchmark using merged model (no PEFT overhead) for fair comparison"""
    model = AutoModelForSeq2SeqLM.from_pretrained(merged_dir, use_safetensors=True)
    model.to(device)
    model.eval()

    lang_token_id = tokenizer.lang_code_to_id["hi_IN"]
    tokenizer.src_lang = "en_XX"

    # Warmup
    for s in samples[:3]:
        inputs = tokenizer(s, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            model.generate(**inputs, forced_bos_token_id=lang_token_id, max_length=64, num_beams=4)

    # Benchmark
    preds = []
    start = time.perf_counter()
    for s in samples:
        inputs = tokenizer(s, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, forced_bos_token_id=lang_token_id, max_length=64, num_beams=4)
        preds.append(tokenizer.decode(out[0], skip_special_tokens=True))
    elapsed = time.perf_counter() - start

    del model
    torch.cuda.empty_cache()
    return preds, elapsed

def benchmark_ct2(ct2_dir, tokenizer, samples, device="cuda"):
    translator = ctranslate2.Translator(ct2_dir, device=device)
    tokenizer.src_lang = "en_XX"

    target_prefix = [tokenizer.convert_ids_to_tokens(tokenizer.lang_code_to_id["hi_IN"])]

    # Warmup - using encode() to include special tokens
    for s in samples[:3]:
        input_ids = tokenizer.encode(s)
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        translator.translate_batch(
            [tokens], target_prefix=[target_prefix],
            beam_size=4, max_decoding_length=64,
        )

    # Benchmark
    preds = []
    start = time.perf_counter()
    for s in samples:
        input_ids = tokenizer.encode(s)
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        result = translator.translate_batch(
            [tokens], target_prefix=[target_prefix],
            beam_size=4, max_decoding_length=64,
        )
        pred_tokens = result[0].hypotheses[0]
        # Remove the language tag from output
        if pred_tokens and pred_tokens[0] == target_prefix[0]:
            pred_tokens = pred_tokens[1:]
        pred = tokenizer.convert_tokens_to_string(pred_tokens)
        preds.append(pred)
    elapsed = time.perf_counter() - start

    return preds, elapsed

def main(args):
    merged_dir = Path(args.merged_dir)
    ct2_dir = Path(args.ct2_dir)

    # Step 1: Merge LoRA
    merge_and_save(args.base_model, args.lora_dir, str(merged_dir))

    # Step 2: Convert to CTranslate2
    convert_to_ct2(str(merged_dir), str(ct2_dir), quantization=args.quantization)

    # Step 3: Size comparison
    lora_size = get_dir_size_mb(args.lora_dir)
    merged_size = get_dir_size_mb(merged_dir)
    ct2_size = get_dir_size_mb(ct2_dir)

    print("\n" + "=" * 50)
    print("MODEL SIZE COMPARISON")
    print("=" * 50)
    print(f"LoRA adapter:     {lora_size:.1f} MB")
    print(f"Merged model:     {merged_size:.1f} MB")
    print(f"CTranslate2:      {ct2_size:.1f} MB")
    print(f"Size reduction:   {(1 - ct2_size / merged_size) * 100:.1f}%")

    # Step 4: Speed benchmark
    # tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))
    tokenizer = MBart50TokenizerFast.from_pretrained(str(merged_dir))

    # Character-spaced inputs (matching training format)
    test_samples = [" ".join(list(s)) for s in [
        "namaste", "kolkata", "chennai", "bharat", "transliteration",
        "gomatinagar", "jhapakana", "bhaskaran", "mumbai", "delhi",
        "bangalore", "hyderabad", "ahmedabad", "lucknow", "jaipur",
        "chandigarh", "bhubaneswar", "thiruvananthapuram", "visakhapatnam", "coimbatore",
    ]]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nBenchmarking on {len(test_samples)} samples ({device})...")

    # Fair comparison: both use merged model
    pt_preds, pt_time = benchmark_pytorch(str(merged_dir), tokenizer, test_samples, device)
    ct2_preds, ct2_time = benchmark_ct2(str(ct2_dir), tokenizer, test_samples, device)

    print("\n" + "=" * 50)
    print("SPEED COMPARISON")
    print("=" * 50)
    print(f"PyTorch (merged):   {pt_time:.3f}s total, {pt_time/len(test_samples)*1000:.1f}ms/sample")
    print(f"CTranslate2 (int8): {ct2_time:.3f}s total, {ct2_time/len(test_samples)*1000:.1f}ms/sample")
    print(f"Speedup:            {pt_time/ct2_time:.2f}x")

    # Quality check
    matches = sum(a.strip() == b.strip() for a, b in zip(pt_preds, ct2_preds))
    print(f"\nPrediction match:   {matches}/{len(test_samples)} ({matches/len(test_samples)*100:.1f}%)")

    print("\nSample outputs:")
    for i in range(min(10, len(test_samples))):
        src = test_samples[i].replace(" ", "")
        match = "✓" if pt_preds[i].strip() == ct2_preds[i].strip() else "✗"
        print(f"  {src}: PyTorch={pt_preds[i]} | CT2={ct2_preds[i]} {match}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_dir", default="outputs/mbart-lora-run-6")
    parser.add_argument("--base_model", default="facebook/mbart-large-50-many-to-many-mmt")
    parser.add_argument("--merged_dir", default="outputs/mbart-merged")
    parser.add_argument("--ct2_dir", default="outputs/mbart-ct2")
    parser.add_argument("--quantization", default="int8", choices=["int8", "float16", "int8_float16"])
    args = parser.parse_args()
    main(args)





########################################


# python Scripts/convert_ct2.py \
#     --lora_dir outputs/mbart-lora-run-6 \
#     --base_model facebook/mbart-large-50-many-to-many-mmt \
#     --merged_dir outputs/mbart-merged \
#     --ct2_dir outputs/mbart-ct2 \
#     --quantization int8