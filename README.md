# Veena Hinglish TTS

Fine-tuning and inference pipeline for Veena TTS model on Hinglish (Hindi-English code-mixed) speech.

## 🚀 Quick Start

### Prerequisites

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file with your API keys:

```env
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_key  # Optional, for OpenAI TTS comparison
```

---

## 📋 Commands Reference

### 1. Training LoRA Adapter

Train a LoRA adapter on your Hinglish dataset:

```bash
python train_veena_lora.py
```

**Configuration** (edit in `train_veena_lora.py`):
- `MODEL_ID`: Base model (default: `maya-research/veena-tts`)
- `DATASET_ID`: HuggingFace dataset ID
- `OUTPUT_DIR`: Where to save checkpoints
- `LR`: Learning rate
- `MAX_SAMPLES`: Limit samples for testing

---

### 2. Merge LoRA Weights

Merge trained LoRA adapter with base model:

```bash
python merge_lora.py
```

**Output**: `./veena_hinglish_merged/`

---

### 3. Run Inference / Evaluation

Generate audio for evaluation samples:

```bash
# Using merged model (default)
python inference_veena.py \
    --model ./veena_hinglish_merged \
    --input eval_data_25.csv \
    --output eval_audio_merged \
    --speaker mixed_hinglish_Speaker

# Using base Veena model
python inference_veena.py \
    --model maya-research/veena-tts \
    --input eval_data_25.csv \
    --output eval_audio_base \
    --speaker kavya

# Using HuggingFace model
python inference_veena.py \
    --model akh99/veena-hinglish-tts \
    --input eval_data_25.csv \
    --output eval_audio_hf

# With 4-bit quantization (saves memory)
python inference_veena.py \
    --model ./veena_hinglish_merged \
    --use-4bit \
    --output eval_audio_4bit
```

**Arguments**:
| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `./veena_hinglish_merged` | Model path or HF ID |
| `--input` | `eval_data_25.csv` | Input CSV file |
| `--output` | `eval_audio_output` | Output directory |
| `--speaker` | `mixed_hinglish_Speaker` | Speaker voice |
| `--temperature` | `0.4` | Sampling temperature |
| `--top-p` | `0.9` | Top-p sampling |
| `--use-4bit` | False | Use 4-bit quantization |
| `--skip-existing` | False | Skip existing files |

---

### 4. Upload Model to HuggingFace Hub

```bash
python upload_model_hf.py \
    --repo-id YOUR_USERNAME/model-name \
    --model-path ./veena_hinglish_merged

# Make it private
python upload_model_hf.py \
    --repo-id YOUR_USERNAME/model-name \
    --model-path ./veena_hinglish_merged \
    --private
```

**Arguments**:
| Argument | Description |
|----------|-------------|
| `--repo-id` | HuggingFace repo (e.g., `akh99/veena-hinglish-tts`) |
| `--model-path` | Local model directory |
| `--private` | Make repository private |
| `--commit-message` | Custom commit message |

---

### 5. Upload Dataset to HuggingFace Hub

```bash
python upload_hf_dataset.py
```

Edit the script to configure:
- `DATASET_NAME`: HuggingFace dataset ID
- `AUDIO_DIR`: Directory with audio files
- `METADATA_FILE`: CSV/JSON with text transcriptions

---

## 📁 Project Structure

```
vee-ana/
├── train_veena_lora.py      # LoRA fine-tuning script
├── merge_lora.py            # Merge LoRA with base model
├── inference_veena.py       # Run inference/evaluation
├── upload_model_hf.py       # Upload model to HF Hub
├── upload_hf_dataset.py     # Upload dataset to HF Hub
├── encode_audio.py          # Audio → SNAC tokens
├── run_inference.py         # Legacy inference script
├── eval_data_25.csv         # 25-sample evaluation set
├── eval_data_50.csv         # 50-sample evaluation set
├── requirements.txt         # Python dependencies
├── .env                     # API keys (not committed)
└── veena_hinglish_merged/   # Merged model output
```

---

## 🔗 Models & Datasets

- **Base Model**: [maya-research/veena-tts](https://huggingface.co/maya-research/veena-tts)
- **Fine-tuned Model**: [akh99/veena-hinglish-tts](https://huggingface.co/akh99/veena-hinglish-tts)
- **Training Dataset**: [akh99/hinglish-tts-openai](https://huggingface.co/datasets/akh99/hinglish-tts-openai)

---

## 📊 Evaluation

The evaluation CSVs contain Hinglish sentences across categories:
- Hindi-dominant
- English-dominant
- Balanced code-mixing
- Technical terms
- Numbers and dates

After generating audio, compare MOS (Mean Opinion Score) between:
1. Base Veena model
2. Fine-tuned Hinglish model
3. OpenAI TTS (reference)
