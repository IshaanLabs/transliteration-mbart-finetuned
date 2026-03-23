# Multilingual Transliteration Model Development: Technical Documentation

## Project Overview

This document presents the systematic development of a multilingual transliteration system capable of converting English text to Hindi, Bengali, and Tamil scripts. The project was undertaken as part of a comprehensive NLP engineering assessment, with the objective of demonstrating proficiency in model training, optimization, and deployment.

### Problem Statement

The task required building a production-ready transliteration system using the Aksharantar dataset, implementing parameter-efficient fine-tuning techniques, optimizing for inference speed using CTranslate2, and deploying an interactive demonstration on HuggingFace Spaces.

### Technical Requirements

- **Dataset**: Aksharantar transliteration corpus for Hindi, Bengali, and Tamil
- **Model Architecture**: Sequence-to-sequence transformer (mBART or mT5)
- **Training Method**: Parameter-efficient fine-tuning using LoRA
- **Optimization**: CTranslate2 for production inference
- **Deployment**: Interactive Gradio application on HuggingFace Spaces
- **Evaluation Metrics**: Exact match accuracy, Character Error Rate (CER), BLEU score

## Phase 1: Data Preparation and Initial Experiments

### Dataset Processing

The Aksharantar dataset presented immediate challenges due to inconsistent schema across language files. The HuggingFace datasets library's default configuration mixed languages and columns inappropriately, necessitating a custom data loading approach.

**Implementation**: Direct zip file downloads from HuggingFace Hub with manual NDJSON parsing, ensuring proper column mapping between "english_word" and "native_word" fields. The preprocessing pipeline included text normalization (lowercase for English, NFC for native scripts) and balanced sampling across languages.

**Results**: Successfully generated clean TSV files with 1.3M Hindi, 1.23M Bengali, and 3.2M Tamil training samples, with corresponding validation sets of 6K-14K samples per language.

### Initial Model Selection: mT5-small

**Rationale**: Given hardware constraints (RTX 3050, 4GB VRAM), mT5-small was selected as the initial baseline due to its multilingual capabilities and manageable parameter count.

**Configuration**:

- Model: `google/mt5-small`
- LoRA parameters: r=16, alpha=32
- Sequence lengths: 48-64 tokens
- Training: batch_size=4, gradient_accumulation=8-16, FP16 precision

**Challenges Encountered**:

1. **Training Instability**: NaN losses appeared early in training
2. **Label Masking Issues**: Padding tokens weren't properly masked (-100)
3. **Evaluation Errors**: Negative label IDs caused decoding failures
4. **Hyperparameter Sensitivity**: High learning rates (1e-4) proved unstable with FP16

**Solutions Applied**:

- Implemented proper label masking for padding tokens
- Reduced learning rate to 3e-5
- Sanitized negative values in evaluation metrics
- Removed problematic notebook callbacks

**Outcome**: Despite fixes, the model exhibited collapsed behavior with 0% accuracy and persistent NaN losses, indicating fundamental training instability.

## Phase 2: Transition to mBART Architecture

### Model Architecture Change

**Rationale**: Transitioned to mBART-50 due to its explicit multilingual design and superior handling of cross-lingual generation tasks.

**Initial Configuration**:

- Model: `facebook/mbart-large-50-many-to-many-mmt`
- LoRA parameters: r=24, alpha=48
- Target modules: ["q_proj", "k_proj", "v_proj", "out_proj"]
- Training: batch_size=16, gradient_accumulation=4

### Critical Language Handling Discovery

**Initial Approach**: Applied T5-style language prefixes (`<hi>`, `<bn>`, `<ta>`) assuming similar behavior across sequence-to-sequence models.

**Problem Identified**: mBART requires explicit language codes (`hi_IN`, `bn_IN`, `ta_IN`) and forced BOS tokens during generation. Simple prefixes resulted in incorrect language steering.

**Results**: Achieved only 2.58% exact match accuracy with 0.699 CER, producing outputs in wrong scripts or unintelligible text.

### Corrected Implementation

**Language Handling Fix**:

- Proper language code mapping: hi→hi_IN, bn→bn_IN, ta→ta_IN
- Source language specification: `tokenizer.src_lang = "en_XX"`
- Target language token prepending in training data
- Forced BOS token implementation for generation

**Training Configuration**:

```python
# Training command used:
python Scripts/train_mbart_lora.py \
    --data_dir data/processed \
    --output_dir outputs/mbart-lora-run-4 \
    --epochs 3 \
    --batch_size 8 \
    --lr 5.6e-5 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05
```

**Evaluation Methodology**: Developed PEFT-aware evaluation script with proper generation settings including beam search and forced BOS tokens.

## Phase 3: Systematic Optimization

### Run 4: Baseline mBART with Correct Language Handling

**Configuration**:

- LoRA: r=16, alpha=32, dropout=0.05
- Target modules: Attention layers only
- Learning rate: 5.6e-5
- Epochs: 3
- Sequence length: 128 tokens

**Results**: Qualitative improvements observed, particularly for Tamil. Hindi and Bengali showed progress but remained suboptimal.

### Run 5: Enhanced Architecture and Character-Level Processing

**Key Innovation - Character Spacing**: Implemented character-level input preprocessing by spacing individual characters (`"namaste" → "n a m a s t e"`). This forces mBART's subword tokenizer to treat each character individually, crucial for transliteration tasks.

**Architecture Expansion**:

- Extended LoRA target modules to include feed-forward layers: ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
- Increased training epochs to 10
- Maintained conservative learning rate (3e-5)

**Training Configuration**:

```python
# Character spacing implementation:
def add_lang(ex):
    spaced = " ".join(list(ex["src"]))
    ex["lang_code"] = LANG_CODE[ex["lang"]]
    ex["input_text"] = spaced
    return ex
```

**Results**:

- Training evaluation: 19.46% accuracy, 0.556 CER
- Standalone evaluation: 22.96% accuracy, 0.300 CER, 57.22 BLEU
- Significant improvement demonstrating the effectiveness of character-level processing

### Run 6: Final Optimization

**Configuration Refinements**:

- Increased LoRA capacity: r=32, alpha=64
- Optimized training duration: 5 epochs (based on convergence analysis)
- Enhanced dropout: 0.1 for better regularization

**Training Command**:

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

**Final Results**:

- Overall accuracy: 28.71%
- Character Error Rate: 0.246
- BLEU Score: 63.94
- Training time: 45 hours and 28 minutes (5 epochs)

**Language-Specific Performance**:

- Tamil: 45.36% accuracy (best performing)
- Hindi: 28.76% accuracy
- Bengali: 12.02% accuracy (challenging due to script complexity)

## Technical Challenges and Solutions

### Challenge 1: Dataset Schema Inconsistency

**Problem**: Aksharantar dataset had varying schemas across language files, preventing standard loading methods.
**Solution**: Implemented custom per-language zip file processing with explicit column mapping and NDJSON parsing.

### Challenge 2: Model-Specific Language Handling

**Problem**: Incorrect assumption that mBART uses T5-style language prefixes.
**Solution**: Researched and implemented mBART's native language code system with proper forced BOS token generation.

### Challenge 3: Transliteration-Specific Tokenization

**Problem**: Standard subword tokenization grouped characters inappropriately for phonetic mapping.
**Solution**: Character spacing preprocessing to force character-level tokenization, dramatically improving transliteration quality.

### Challenge 4: Evaluation Methodology

**Problem**: Training-time evaluation with teacher forcing didn't reflect generation performance.
**Solution**: Developed standalone evaluation with beam search, proper language steering, and production-equivalent generation settings.

## Design Decisions and Rationale

### Model Selection: mBART over mT5

**Rationale**: mBART's explicit multilingual architecture and built-in language handling capabilities made it superior for cross-lingual generation tasks compared to mT5's prefix-based approach.

### Parameter-Efficient Fine-Tuning: LoRA

**Benefits**:

- Reduced computational requirements (fits consumer hardware)
- Faster training convergence
- Lower risk of overfitting on limited data
- Modular adaptation allowing base model preservation

### Character-Level Processing

**Innovation**: Character spacing preprocessing emerged as the most impactful optimization, improving accuracy by 7+ percentage points. This domain-specific insight proved more valuable than generic hyperparameter tuning.

### Target Module Selection

**Evolution**: Progressed from attention-only LoRA to including feed-forward layers, providing additional capacity for character-level mapping without excessive parameter growth.

## Performance Analysis

### Quantitative Results

| Iteration    | Configuration           | Exact Match | CER     | BLEU    | Key Changes          |
| ------------ | ----------------------- | ----------- | ------- | ------- | -------------------- |
| mT5 Baseline | r=16, unstable training | ~0%         | ~2.0    | N/A     | Collapsed model      |
| mBART Run 3  | Wrong lang handling     | 2.58%       | 0.699   | N/A     | Architecture change  |
| mBART Run 4  | Correct lang codes      | Pending     | Pending | Pending | Language fix         |
| mBART Run 5  | Character spacing       | 22.96%      | 0.300   | 57.22   | Character processing |
| mBART Run 6  | Optimized LoRA          | 28.71%      | 0.246   | 63.94   | Final optimization   |

### Qualitative Observations

**Script Complexity Impact**: Performance varied significantly by target language, correlating with orthographic complexity:

- Tamil (simple character-sound mapping): 45.36%
- Hindi (moderate complexity with conjuncts): 28.76%
- Bengali (high complexity with contextual variations): 12.02%

**Training vs. Evaluation Gap**: Standalone evaluation consistently outperformed training metrics due to beam search vs. greedy decoding and proper language token handling.

## Model Optimization and Deployment

### CTranslate2 Optimization

**Process**:

```bash
python Scripts/convert_ct2.py \
    --lora_dir outputs/mbart-lora-run-6 \
    --base_model facebook/mbart-large-50-many-to-many-mmt \
    --merged_dir outputs/mbart-merged \
    --ct2_dir outputs/mbart-ct2 \
    --quantization int8
```

**Optimization Results**:

- Model size reduction: 74.5% (2347MB → 598MB)
- Inference speedup: 2.66x (51ms → 19ms per sample)
- Quality retention: 95% prediction match
- Memory efficiency: Significant reduction in runtime requirements

### Production Deployment

**Gradio Application**: Developed interactive web interface with real-time transliteration, hardware auto-detection (GPU/CPU), and comprehensive language support.

**HuggingFace Spaces**: Successfully deployed optimized model with zero-setup user access and automatic scaling capabilities.

## Potential Improvements

### Technical Enhancements

1. **Bengali-Specific Processing**: Implement script-specific preprocessing for complex conjunct consonant handling
2. **Dataset Balancing**: Address training data imbalance across languages
3. **Advanced Decoding**: Explore constrained decoding and phonetic similarity scoring
4. **Contextual Transliteration**: Handle ambiguous cases with surrounding context

### Methodological Improvements

1. **Evaluation Metrics**: Develop phonetically-aware evaluation beyond exact match
2. **Cross-Validation**: Implement robust evaluation across multiple data splits
3. **Error Analysis**: Systematic categorization of failure modes by language and word type

## Conclusion

This project successfully developed a multilingual transliteration system achieving 28.71% overall accuracy through systematic experimentation and optimization. Key contributions include:

1. **Character-level processing innovation** for improved transliteration quality
2. **Proper multilingual model handling** for mBART architecture
3. **Production-ready optimization** using CTranslate2
4. **Comprehensive evaluation methodology** reflecting real-world performance

The systematic approach from initial failures with mT5 through optimized mBART implementation demonstrates the importance of understanding model-specific requirements and domain-appropriate preprocessing techniques in multilingual NLP applications.
