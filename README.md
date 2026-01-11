# Veena Hinglish TTS

Fine-tuned Veena TTS model for Hinglish (Hindi-English code-mixed) speech synthesis.

**Performance**: Base model MOS: 4.12/5 → Fine-tuned model: **4.66/5** ⭐

## 🚀 Quick Start

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

## 📊 Training & Datasets

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

### Best Model: `veena-hinglish-tts` ⭐

| Model | Based On | Performance |
|-------|----------|-------------|
| **veena-hinglish-tts** | Indic TTS → Hinglish | **4.66/5 MOS (Best)** |
| veena-hinglish | OpenAI-generated Hinglish | Alternative variant |
| hinglish-tts-akhila | Eleven Labs-generated Hinglish | Alternative variant |

**Note**: The best model uses authentic Hindi speech as foundation, resulting in superior quality.

---

## 🎯 Workflow

### 1️⃣ Train Model

```bash
jupyter notebook train.ipynb
```

Trains a LoRA adapter on Hinglish dataset.

### 2️⃣ Merge Adapter

```bash
python merge_lora.py
```

Merges LoRA weights with base Veena model → `./veena_hinglish_merged/`

### 3️⃣ Upload to HuggingFace

```bash
python upload_model_hf.py --repo-id akh99/veena-hinglish-tts --model-path ./veena_hinglish_merged
```

### 4️⃣ Stream Audio (Real-time Inference)

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

### 5️⃣ Evaluate Model

Run MOS evaluation on `eval_data_25.csv`:

```bash
python eval_mos.py --model akh99/veena-hinglish-tts --use-4bit
```

**Note**: Evaluation uses **4-bit quantization** to manage memory while maintaining quality.

---

## 📁 Project Files

### Training & Inference
| File | Purpose |
|------|---------|
| `train.ipynb` | LoRA fine-tuning notebook |
| `merge_lora.py` | Merge LoRA adapter with base model |
| `upload_model_hf.py` | Upload trained model to HuggingFace |

### Streaming (Real-time)
| File | Purpose |
|------|---------|
| `streaming.py` | FastAPI server (vLLM + SNAC decoder, uses bf16) |
| `streaming_client.py` | Streamlit client UI for streaming |

### Audio Generation (Offline)
| File | Purpose |
|------|---------|
| `generate_audio_batch_veena.py` | Generate audio using Veena model |
| `generate_11_labs.py` | Generate audio using Eleven Labs TTS |
| `generate_openai.py` | Generate audio using OpenAI TTS |
| `batch_generate_audio.py` | Batch processing utility |

### Evaluation & Utils
| File | Purpose |
|------|---------|
| `eval_data_25.csv` | 25 Hinglish sentences for evaluation (used for MOS) |
| `eval_50.ipynb` | Evaluation notebook for 50-sample set |
| `utils.py` | Helper functions (OpenAI TTS integration, etc.) |

### Data & Preprocessing
| File | Purpose |
|------|---------|
| `hinglish_texts.json` | LLM-generated Hinglish transliterations from Indic TTS dataset |
| `hinglish_transliterated.txt` | Hindi→English transliterations from `mixed_code.txt` (preprocessing step) |
| `mixed_code.txt` | Original mixed code (Hindi + English words) |

---

## 📈 Evaluation Results

**MOS (Mean Opinion Score) Evaluation**:
- Base Veena model: **4.12/5**
- Fine-tuned Hinglish model: **4.66/5** ⭐
- **Improvement**: +0.54 points (+13% relative improvement)

**Technical Notes**:
- **Streaming**: Uses bfloat16 precision (faster inference with minimal quality loss)
- **Evaluation**: Uses 4-bit quantization (memory efficient for MOS scoring)
- **Evaluation methodology**: Subjective quality assessment with efforts to minimize bias

---

## 🔗 Models & Datasets

- **Base Model**: [maya-research/veena-tts](https://huggingface.co/maya-research/veena-tts)
- **Fine-tuned Model**: [akh99/veena-hinglish-tts](https://huggingface.co/akh99/veena-hinglish-tts) ⭐
- **Training Dataset**: [akh99/hinglish-tts-openai](https://huggingface.co/datasets/akh99/hinglish-tts-openai)

---

## 📝 File Examples

### `mixed_code.txt` (Original)
```
मेरे पास एक dog है।
आपका name क्या है?
```

### `hinglish_transliterated.txt` (After preprocessing)
```
Mere paas ek dog hai.
Aapka name kya hai?
```

### `hinglish_texts.json` (Indic TTS conversion)
```json
{
  "hindi_text": "नमस्ते, आप कैसे हैं?",
  "hinglish_text": "Namaste, aap kaise ho?",
  "source": "indic_tts_dataset"
}
```

---

## 🙏 Acknowledgments

- Base model: [Maya Research - Veena TTS](https://github.com/maya-research/veena)
- Audio codec: [SNAC](https://github.com/hubertsiuzdak/snac)
- Inference: [vLLM](https://vllm.ai/)
