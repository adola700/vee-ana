# Technical Details & Documentation

This document contains in-depth information about the data sources, training process, and file structure for Veena Hinglish TTS.

---

## Models & Datasets

- **Base Model**: [maya-research/veena-tts](https://huggingface.co/maya-research/veena-tts)
- **Fine-tuned Model**: [akh99/veena-hinglish-stage1](https://huggingface.co/akh99/veena-hinglish-stage1) ⭐ (Best)
- **Training Dataset**: [akh99/indictts-hinglish](https://huggingface.co/datasets/akh99/indictts-hinglish)

---

## Data Sources & Training Details

### Speaker Management

**Important Design Decision**: To preserve the original speaker identity (e.g., "kavya"), during training we ensured that each dataset used a different speaker name based on its source:

-   Default speaker across inference scripts: **"kavya"** (Veena base model's default voice)
-   During training: Custom speaker names were assigned per dataset to prevent overwriting the original speaker
-   This prevents the fine-tuned model from modifying or conflicting with the base model's native speaker voices

**Default Speaker Configuration**:
```python
speaker = "kavya"  # Default in all inference scripts
```

When you run inference with `run_inference.py` or evaluation scripts, the output will use the "kavya" voice unless explicitly modified.

### Data Sources

We created Veena Hinglish TTS using 2 primary data sources:

#### **Data Source 1: Generated Hinglish (2000 utterances)**
-   Mixed Hindi-English code-switching sentences
-   TTS-generated audio:
    -   **Eleven Labs V3**: [akh99/hinglish-tts-akhila](https://huggingface.co/datasets/akh99/hinglish-tts-akhila)
    -   **GPT-4 Mini TTS**: [akh99/hinglish-tts-openai](https://huggingface.co/datasets/akh99/hinglish-tts-openai)
-   **Quality boost**: Converted English transliterations to actual Hindi script using LLM, improving pronunciation naturalness

#### **Data Source 2: Indic TTS (Hindi corpus) → Hinglish**
-   **Source**: Obtained from `SPRINGLab/IndicTTS-Hindi`
-   **Structure**: Hinglish column added to suit our requirements
-   **Method**: Hinglish column generated using LLM call; other columns taken directly from IndicTTS dataset


### Best Model Comparison

| Model | Based On | Performance |
|-------|----------|-------------|
| **veena-hinglish-stage1** ⭐ | Indic TTS → Hinglish | **4.66/5 MOS (Best)** |
| veena-hinglish-tts | openai dataset - akh99/hinglish-tts-openai | Alternative variant |
| hinglish-tts-akhila | Eleven Labs V3: akh99/hinglish-tts-akhila | Alternative variant |

**Why the best model outperforms others:**
-   Best model: Trained on **authentic Hindi speech** (Indic TTS dataset)
-   Best model: Large-scale dataset converted to Hinglish
-   Other models: Trained on **artificially-generated TTS audio** (OpenAI/Eleven Labs)
-   Other models: **~20x smaller dataset** (only 2000 synthetic utterances vs. large Indic corpus)
-   Result: Superior naturalness and pronunciation quality due to real human speech foundation

**Note**: The best model uses authentic Hindi speech as foundation, resulting in superior quality compared to models trained on synthetic TTS data.

### Evaluation Results

**MOS (Mean Opinion Score) on eval_data_25.csv**:
-   Base Veena model: **4.12/5**
-   Fine-tuned Hinglish model: **4.66/5** ⭐
-   **Improvement**: +0.54 points (+13% relative improvement)

**Technical Notes**:


-   **Evaluation methodology**: Subjective quality assessment with efforts to minimize bias

---

## Detailed Data Files & Audio Generation

### Data & Preprocessing Files

| File | Purpose | How to Use |
|------|---------|-----------|
| `hinglish_texts.json` | LLM-generated Hinglish transliterations from IndicTTS dataset | Reference for dataset conversion |
| `hinglish_transliterated.txt` | Devanagari script transliterations from `mixed_code.txt` | Output of conversion (used for TTS training) |
| `mixed_code.txt` | Original mixed code in Roman script | Input to LLM for transliteration to Devanagari |
| `process_indictts_hinglish.py` | Processes IndicTTS dataset using the JSON mapping | `python process_indictts_hinglish.py` (Loads dataset, matches Hinglish, uploads to HF) |

### Audio Generation (Offline Batch Processing)

| File | Purpose | How to Use |
|------|---------|-----------|
| `run_inference.py` | Generate audio in batch using Veena model (Offline) | `python run_inference.py` (reads `eval_data_25.csv` by default) |
| `generate_11_labs.py` | Generate audio using Eleven Labs TTS API | Setup API key, run for batch generation |
| `generate_openai.py` | Generate audio using OpenAI TTS mini model | Setup API key in `utils.py`, run for batch |
| `batch_generate_audio.py` | Batch processing utility for OpenAI TTS | `python batch_generate_audio.py --input <file> --output-dir <dir>` |

### Data Transformation Example

```
mixed_code.txt (Original Roman Hinglish)
└─→ LLM Processing (convert Roman to Devanagari script)
    └─→ hinglish_transliterated.txt (Devanagari Script)
    
hinglish_texts.json (Indic TTS dataset conversion)
└─→ LLM Processing (Hindi → Hinglish)
    └─→ training data for models
```

### Preprocessing Example

**`mixed_code.txt` (Original Roman Hinglish)**
```
Mere paas ek dog hai.
Aapka name kya hai?
```

**`hinglish_transliterated.txt` (After transliteration to Devanagari)**
```
मेरे पास एक dog है।
आपका name क्या है?
```

**`hinglish_texts.json` (Indic TTS → Hinglish conversion)**
```json
{
  "hindi_text": "नमस्ते, आप कैसे हैं?",
  "hinglish_text": "Namaste, aap kaise ho?",
  "source": "indic_tts_dataset"
}
```
