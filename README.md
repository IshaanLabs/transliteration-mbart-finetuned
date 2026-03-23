# Multilingual Transliteration Model: English to Indic Scripts

A transliteration system that converts English text to Hindi, Bengali, and Tamil using fine-tuned mBART-50 with LoRA optimization and CTranslate2 acceleration.

## 🎯 Project Overview

This project tackles the challenge of accurate English-to-Indic script transliteration using modern NLP techniques. By fine-tuning Facebook's mBART-50 model with LoRA (Low-Rank Adaptation) on the Aksharantar dataset, we achieved a lightweight yet effective transliteration system optimized for production deployment.

**Live Demo**: [🚀 Try it on HuggingFace Spaces](https://huggingface.co/spaces/IshaanLabs/transliteration-mbart-finetuned)

## 📊 Key Results

- **Overall Accuracy**: 28.71% (exact match)
- **Character Error Rate**: 0.246
- **BLEU Score**: 63.94
- **Model Size Reduction**: 74.5% (2.3GB → 598MB)
- **Inference Speedup**: 2.66x faster with CTranslate2
- **Training Time**: ~45 hours and 28 minutes (5 epochs)

### Language-Specific Performance

| Language | Accuracy | Sample Input → Output  |
| -------- | -------- | ----------------------- |
| Tamil    | 45.36%   | chennai → சென்னை |
| Hindi    | 28.76%   | namaste → नमस्ते |
| Bengali  | 12.02%   | kolkata → কলকাতা |

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM
- 5GB free disk space

### Environment Setup

1. **Clone the repository**

```bash
git clone https://github.com/IshaanLabs/transliteration-mbart-finetuned.git
```

2. **Create virtual environment**

```bash
python -m venv env
# Windows
env\Scripts\activate
# Linux/Mac
source env/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Required Dependencies

The project uses these key packages:

- `torch>=2.5.1` - PyTorch framework
- `transformers>=5.3.0` - HuggingFace transformers
- `peft>=0.18.1` - Parameter-Efficient Fine-Tuning
- `ctranslate2>=4.0.0` - Model optimization
- `datasets>=4.6.1` - Dataset handling
- `gradio>=4.0.0` - Web interface

See [requirements.txt](requirements.txt) for complete list.

## 📚 Dataset Preparation

The Aksharantar dataset contains transliteration pairs for multiple Indic languages. Due to schema inconsistencies in the HuggingFace dataset, we implemented a custom data loader.

```bash
python Scripts/prepare_data.py
```

This script:

- Downloads individual language files from HuggingFace
- Processes train/validation splits
- Handles schema variations across languages
- Outputs clean TSV files in `data/processed/`

**Languages processed**: Hindi (`hin`), Bengali (`ben`), Tamil (`tam`)

## 🚀 Training Process

### Model Architecture

- **Base Model**: `facebook/mbart-large-50-many-to-many-mmt`
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Target Modules**: Attention layers + Feed-forward networks
- **Key Innovation**: Character spacing preprocessing

### Training Configuration

```bash
python Scripts/train_mbart_lora.py \
    --data_dir data/processed \
    --output_dir outputs/mbart-lora-run-6 \
    --epochs 5 \
    --batch_size 8 \
    --lr 3e-5 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --max_source_length 128 \
    --max_target_length 64
```

### Hyperparameters

| Parameter     | Value | Description                             |
| ------------- | ----- | --------------------------------------- |
| Learning Rate | 3e-5  | Optimized for LoRA fine-tuning          |
| LoRA Rank (r) | 32    | Balance between efficiency and capacity |
| LoRA Alpha    | 64    | Scaling factor for adaptation           |
| Dropout       | 0.1   | Regularization                          |
| Batch Size    | 8     | Per-device batch size                   |
| Epochs        | 5     | Optimal convergence point               |

### Training Insights

**Character Spacing**: The breakthrough came from preprocessing input text with character spacing (`" ".join(list(text))`). This forces mBART's subword tokenizer to treat each character individually, which is crucial for transliteration tasks.

**LoRA Configuration**: We target both attention mechanisms and feed-forward networks:

```python
target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
```

**Training Time**: 45 hours and 28 minutes minutes (5 epochs), significantly faster than full fine-tuning while maintaining quality.

## 📈 Evaluation Results

### Standalone Evaluation

```bash
python Scripts/eval_lora_mabart_run1.py \
    --model_dir outputs/mbart-lora-run-6 \
    --base_model facebook/mbart-large-50-many-to-many-mmt \
    --data_dir data/processed \
    --max_eval_samples 5000
```

**Results**:

- Exact Match Accuracy: 28.71%
- Character Error Rate: 0.246
- BLEU Score: 63.94

### Training vs. Evaluation Metrics

| Metric Type | Training Eval | Standalone Eval | Difference |
| ----------- | ------------- | --------------- | ---------- |
| Accuracy    | 19.46%        | 28.71%          | +9.25%     |
| CER         | 0.556         | 0.246           | -0.310     |

The significant improvement in standalone evaluation comes from:

- Beam search decoding (vs. greedy during training)
- Proper language token handling
- No padding artifacts in generation

## ⚡ CTranslate2 Optimization

### Conversion Process

```bash
python Scripts/convert_ct2.py \
    --lora_dir outputs/mbart-lora-run-6 \
    --base_model facebook/mbart-large-50-many-to-many-mmt \
    --merged_dir outputs/mbart-merged \
    --ct2_dir outputs/mbart-ct2 \
    --quantization int8
```

### Benchmarking Results

| Metric                   | PyTorch Model | CTranslate2 Model | Improvement               |
| ------------------------ | ------------- | ----------------- | ------------------------- |
| **Model Size**     | 2,347 MB      | 598 MB            | **74.5% reduction** |
| **Inference Time** | 51 ms/sample  | 19 ms/sample      | **2.66x speedup**   |
| **Memory Usage**   | High          | Low               | Significant               |
| **Quality Match**  | 100%          | 95%               | Minimal loss              |

### Quality Validation

Out of 20 test samples, 19 predictions matched exactly (95% agreement). The single mismatch was "delhi" with both outputs being valid Hindi transliterations, demonstrating that the optimization preserves model quality.

## 🎮 Sample Outputs

### Hindi (Devanagari)

```
Input: namaste    → Output: नमस्ते
Input: delhi      → Output: दिल्ली
Input: bharat     → Output: भारत
Input: yoga       → Output: योग
```

### Bengali

```
Input: kolkata    → Output: কলকাতা
Input: bangladesh → Output: বাংলাদেশ
Input: dhaka      → Output: ঢাকা
Input: bengali    → Output: বাঙালি
```

### Tamil

```
Input: chennai    → Output: சென்னை
Input: madurai    → Output: மதுரை
Input: tamil      → Output: தமிழ்
Input: bangalore  → Output: பெங்களூர்
```

## 🌐 Interactive Demo & Deployment

### 🎮 Try the Live Demo

We've built an intuitive Gradio web application that lets you experience the transliteration model instantly:

**🚀 Live Demo**: [Try it on HuggingFace Spaces](https://huggingface.co/spaces/IshaanLabs/transliteration-mbart-finetuned)

### Demo Features

- **Real-time transliteration** with live inference timing
- **Multi-language selection** (Hindi, Bengali, Tamil)
- **Smart hardware detection** (automatically uses GPU when available, falls back to CPU)
- **Example inputs** to get you started quickly
- **Clean, intuitive interface** suitable for both technical and non-technical users
- **Device performance info** shows whether GPU or CPU was used

### Running the Demo Locally

The complete Gradio application is included in this repository under the `application/` directory:

```bash
cd application
pip install -r requirements.txt
python app.py
```

The demo will be available at `http://localhost:7860`

### How to Use the Demo

1. **Enter English text** (e.g., "namaste", "kolkata", "chennai")
2. **Select target language** from Hindi, Bengali, or Tamil
3. **Click Submit** and see the transliterated result
4. **Check timing info** to see inference speed and hardware used

### Production Deployment

The demo is deployed on HuggingFace Spaces using the CTranslate2 optimized model:

- **No installation required** - just click and use
- **Fast inference** even on CPU-only infrastructure
- **Mobile-friendly** responsive interface
- **Free access** for everyone to try

## 📁 Project Structure

```
transliteration-mbart-finetuned/
├── All_Scripts/                    # Complete collection of scripts
│   ├── convert_ct2.py             # CTranslate2 conversion
│   ├── eval_lora_mabart_run.py    # Model evaluation (main)
│   ├── eval_lora.py               # Alternative evaluation script
│   ├── prepare_data.py            # Dataset preparation
│   ├── rectified_prepare_data.py  # Updated data preparation
│   ├── train_mbart_lora_v2.py     # Enhanced training script
│   ├── train_mbart_lora.py        # Original LoRA fine-tuning
│   ├── train.py                   # Base training script
│   └── training.log               # Training logs
├── Scripts/                       # Core production scripts
│   ├── convert_ct2.py             # CTranslate2 conversion
│   ├── eval_lora_mabart_run.py    # Model evaluation
│   ├── prepare_data.py            # Dataset preparation
│   └── train_mbart_lora.py        # LoRA fine-tuning
├── data/                          # Dataset storage
│   └── processed/                 # Language-specific processed data
│       ├── bn/                    # Bengali data files
│       │   ├── test.tsv
│       │   ├── train.tsv
│       │   └── val.tsv
│       ├── hi/                    # Hindi data files
│       │   ├── test.tsv
│       │   ├── train.tsv
│       │   └── val.tsv
│       └── ta/                    # Tamil data files
│           ├── test.tsv
│           ├── train.tsv
│           └── val.tsv
├── models/                        # Optimized model storage
│   └── mbart-ct2/                 # CTranslate2 optimized model
│       ├── config.json
│       ├── model.bin
│       └── shared_vocabulary.json
├── outputs/                       # Training outputs
│   └── mbart-lora-run-6/         # LoRA checkpoint directory
│       ├── checkpoint-*/          # Training checkpoints (3750, 7500, 11250, 15000, 18750)
│       ├── adapter_config.json    # LoRA adapter configuration
│       ├── adapter_model.safetensors # Final LoRA weights
│       ├── tokenizer_config.json  # Tokenizer configuration
│       ├── tokenizer.json         # Tokenizer vocabulary
│       └── training_args.bin      # Training arguments
├── app.py                         # Gradio web application
├── JOURNEY.md                     # Development journey documentation
├── README.md                      # Project documentation
└── requirements.txt               # Project dependencies

```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **AI4Bharat** for the comprehensive Aksharantar dataset
- **Facebook AI Research** for the mBART-50 multilingual model
- **Microsoft Research** for the LoRA fine-tuning technique
- **OpenNMT Team** for the CTranslate2 optimization framework
- **HuggingFace** for the transformers library and Spaces platform

---
