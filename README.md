# Veena Hinglish TTS

Fine-tuned Veena TTS model for Hinglish speech synthesis.

**Performance**: Base model MOS: 4.12/5 → Fine-tuned model: **4.66/5** ⭐

## Quick Start

### Prerequisites

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set up API keys in `.env`:
```env
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_key  # Optional
```

---

## Main Workflow - How to Use

### 1. Train Model

Open and run all cells in the training notebook:

```bash
jupyter notebook train.ipynb
```

Alternatively, execute the notebook from command line:

```bash
jupyter nbconvert --to notebook --execute train.ipynb
```

This trains a LoRA adapter on the Hinglish dataset.

### 2. Merge Adapter

```bash
python merge_lora.py
```

Merges LoRA weights with base Veena model → `./veena_hinglish_merged/`

### 3. Upload to HuggingFace Hub

```bash
python upload_model_hf.py --repo-id akh99/veena-hinglish-stage1 --model-path ./veena_hinglish_merged
```

### 4. Stream Audio (Real-time Inference)

#### Terminal 1 (GPU environment):
```bash
python streaming.py
```
Starts FastAPI server with vLLM (uses **bfloat16** for speed).

#### Terminal 2 (Local machine):
```bash
streamlit run streaming_client.py
```
Interactive web UI for real-time streaming audio.

### 5. Evaluate Model

Evaluation on `eval_data_25.csv` using MOS (Mean Opinion Score):

```bash
python eval_mos.py --model akh99/veena-hinglish-stage1 --use-4bit
```

**Note**: Evaluation uses **4-bit quantization** to manage memory while maintaining quality.

---

## Main Project Files

### Core Training & Serving
| File | How to Use |
|------|-----------|
| `train.ipynb` | Open in Jupyter, edit dataset config, run cells to train LoRA adapter |
| `merge_lora.py` | Run after training to merge LoRA weights with base model |
| `streaming.py` | Run in GPU environment: `python streaming.py` → FastAPI server on port 8000 |
| `streaming_client.py` | Run locally: `streamlit run streaming_client.py` → Interactive UI for streaming audio |
| `upload_model_hf.py` | Upload trained model to HuggingFace Hub after merging |

### Evaluation
| File | How to Use |
|------|-----------|
| `eval_data_25.csv` | 25 test sentences for MOS evaluation (Hindi-English code-mixed) |
| `eval_50.ipynb` | Notebook for 50-sample evaluation analysis |
| `generated_audio/` | Output folder containing final speech MP3s generated from the fine-tuned model on eval_data_25.csv |

### Utilities
| File | Purpose |
|------|---------|
| `utils.py` | Helper functions (OpenAI TTS API calls, etc.) |
| `requirements.txt` | All Python dependencies |

---

## Models & Datasets

- **Base Model**: [maya-research/veena-tts](https://huggingface.co/maya-research/veena-tts)
- **Fine-tuned Model**: [akh99/veena-hinglish-stage1](https://huggingface.co/akh99/veena-hinglish-stage1) ⭐ (Best)
- **Training Dataset**: [akh99/hinglish-tts-openai](https://huggingface.co/datasets/akh99/hinglish-tts-openai)

---

## Extras: Data Sources & Training Details

### Speaker Management

**Important Design Decision**: To preserve the original speaker identity (e.g., "kavya"), during training we ensured that each dataset used a different speaker name based on its source:

- Default speaker across inference scripts: **"kavya"** (Veena base model's default voice)
- During training: Custom speaker names were assigned per dataset to prevent overwriting the original speaker
- This prevents the fine-tuned model from modifying or conflicting with the base model's native speaker voices

**Default Speaker Configuration**:
```python
speaker = "kavya"  # Default in all inference scripts
```

When you run inference with `run_inference.py`, `streaming.py`, or evaluation scripts, the output will use the "kavya" voice unless explicitly modified.

### Data Sources

We created Veena Hinglish TTS using 2 primary data sources:

#### **Data Source 1: Generated Hinglish (2000 utterances)**
- Mixed Hindi-English code-switching sentences
- TTS-generated audio:
  - **Eleven Labs V3**: 3rd dataset variant
  - **GPT-4 Mini TTS**: 1st dataset variant
- **Quality boost**: Converted English transliterations to actual Hindi script using LLM, improving pronunciation naturalness

#### **Data Source 2: Indic TTS (Hindi corpus) → Hinglish**
- Large-scale authentic Hindi speech
- Converted to Hinglish (code-mixed format) to align with real-world usage
- Preserves original audio quality while creating code-mixed training data

### Best Model Comparison

| Model | Based On | Performance |
|-------|----------|-------------|
| **veena-hinglish-stage1** ⭐ | OpenAI-generated Hinglish | **4.66/5 MOS (Best)** |
| veena-hinglish-tts | Indic TTS → Hinglish | Alternative variant |
| hinglish-tts-akhila | Eleven Labs-generated Hinglish | Alternative variant |

**Why the best model outperforms others:**
- Best model: Trained on **authentic Hindi speech** (Indic TTS dataset)
- Best model: Large-scale dataset converted to Hinglish
- Other models: Trained on **artificially-generated TTS audio** (OpenAI/Eleven Labs)
- Other models: **~20x smaller dataset** (only 2000 synthetic utterances vs. large Indic corpus)
- Result: Superior naturalness and pronunciation quality due to real human speech foundation

**Note**: The best model uses authentic Hindi speech as foundation, resulting in superior quality compared to models trained on synthetic TTS data.

### Evaluation Results

**MOS (Mean Opinion Score) on eval_data_25.csv**:
- Base Veena model: **4.12/5**
- Fine-tuned Hinglish model: **4.66/5** ⭐
- **Improvement**: +0.54 points (+13% relative improvement)

**Technical Notes**:
- **Streaming**: Uses bfloat16 precision (faster inference with minimal quality loss)
- **Evaluation**: Uses 4-bit quantization (memory efficient for MOS scoring)
- **Evaluation methodology**: Subjective quality assessment with efforts to minimize bias

---

## Extras: Data Files & Audio Generation

### Data & Preprocessing Files

| File | Purpose | How to Use |
|------|---------|-----------|
| `hinglish_texts.json` | LLM-generated Hinglish transliterations from Indic TTS dataset | Reference for dataset conversion |
| `hinglish_transliterated.txt` | Hindi→English transliterations from `mixed_code.txt` (preprocessing) | Intermediate output from conversion |
| `mixed_code.txt` | Original mixed code with Hindi + English words | Input to LLM for transliteration |

### Audio Generation (Offline Batch Processing)

| File | Purpose | How to Use |
|------|---------|-----------|
| `generate_audio_batch_veena.py` | Generate audio in batch using Veena model | Edit config, run to generate multiple audio files |
| `generate_11_labs.py` | Generate audio using Eleven Labs TTS API | Setup API key, run for batch generation |
| `generate_openai.py` | Generate audio using OpenAI TTS mini model | Setup API key in `utils.py`, run for batch |
| `batch_generate_audio.py` | Batch processing utility for any TTS | Configure input CSV and run for large-scale generation |

### Data Transformation Example

```
mixed_code.txt (Original)
└─→ LLM Processing (convert Hindi to English transliteration)
    └─→ hinglish_transliterated.txt (English transliteration)
    
hinglish_texts.json (Indic TTS dataset conversion)
└─→ LLM Processing (Hindi → Hinglish)
    └─→ training data for models
```

### Preprocessing Example

**`mixed_code.txt` (Original)**
```
मेरे पास एक dog है।
आपका name क्या है?
```

**`hinglish_transliterated.txt` (After transliteration)**
```
Mere paas ek dog hai.
Aapka name kya hai?
```

**`hinglish_texts.json` (Indic TTS → Hinglish conversion)**
```json
{
  "hindi_text": "नमस्ते, आप कैसे हैं?",
  "hinglish_text": "Namaste, aap kaise ho?",
  "source": "indic_tts_dataset"
}
```

---

## Acknowledgments

- Base model: [Maya Research - Veena TTS](https://github.com/maya-research/veena)
- Audio codec: [SNAC](https://github.com/hubertsiuzdak/snac)
- Inference: [vLLM](https://vllm.ai/)
