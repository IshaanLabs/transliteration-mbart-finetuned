import time
import torch
import ctranslate2
import gradio as gr
from transformers import MBart50TokenizerFast

MODEL_DIR = "models/mbart-ct2"
BASE_MODEL = "facebook/mbart-large-50-many-to-many-mmt"

LANG_MAP = {
    "Hindi": "hi_IN",
    "Bengali": "bn_IN",
    "Tamil": "ta_IN",
}

# Auto-detect device and compute type
if torch.cuda.is_available():
    device = "cuda"
    compute_type = "float16"  # GPU supports float16 for better performance
    print(f"Using GPU: {torch.cuda.get_device_name()}")
else:
    device = "cpu"
    compute_type = "int8"  # CPU uses int8 for efficiency
    print("Using CPU")

translator = ctranslate2.Translator(MODEL_DIR, device=device, compute_type=compute_type)
tokenizer = MBart50TokenizerFast.from_pretrained(BASE_MODEL)


def transliterate(text: str, target_lang: str) -> tuple[str, str]:
    if not text.strip():
        return "", ""

    lang_code = LANG_MAP[target_lang]
    tokenizer.src_lang = "en_XX"

    # Character spacing — key trick for transliteration with mBART
    spaced = " ".join(list(text.strip()))

    # Tokenize for CT2: encode → convert to tokens (preserves special tokens)
    input_ids = tokenizer.encode(spaced)
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    target_prefix = [tokenizer.convert_ids_to_tokens(tokenizer.lang_code_to_id[lang_code])]

    start = time.perf_counter()
    results = translator.translate_batch(
        [input_tokens],
        target_prefix=[target_prefix],
        beam_size=4,
        max_decoding_length=128,
    )
    elapsed = time.perf_counter() - start

    output_tokens = results[0].hypotheses[0]
    output_ids = tokenizer.convert_tokens_to_ids(output_tokens)
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return output_text, f"{elapsed * 1000:.1f} ms ({device.upper()})"


demo = gr.Interface(
    fn=transliterate,
    inputs=[
        gr.Textbox(label="English Text", placeholder="e.g. namaste, kolkata, chennai"),
        gr.Radio(choices=list(LANG_MAP.keys()), label="Target Language", value="Hindi"),
    ],
    outputs=[
        gr.Textbox(label="Transliteration"),
        gr.Textbox(label="Inference Time"),
    ],
    title="🔤 English → Indic Transliteration",
    description=(
        "Transliterate English text to Hindi, Bengali, or Tamil using "
        "mBART-50 fine-tuned with LoRA on the Aksharantar dataset, "
        "optimized with CTranslate2 (int8)."
    ),
    examples=[
        ["namaste", "Hindi"],
        ["kolkata", "Bengali"],
        ["chennai", "Tamil"],
        ["delhi", "Hindi"],
        ["mumbai", "Bengali"],
        ["madurai", "Tamil"],
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()