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

Execute the training notebook:

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



### 4. Running Offline Batch Inference (For Evaluation)

To generate audio files for all sentences in a csv file (like `eval_data_25.csv`) using the fine-tuned model:

```bash
python run_inference.py --model_name akh99/veena-hinglish-stage1
```

This will create a `generated_audio/` directory containing the .mp3 files.

### 5. Evaluate Model

Refer to `eval_data_25.csv` for evaluation sentences and `generated_audio/` for the generated speech outputs.

### 6. Real-time Streaming

**Step 1: Start Server (GPU Environment)**
```bash
python streaming.py
```

**Step 2: Start Client (Local Machine)**
```bash
python streaming_client.py
```
---

## Main Project Files

### Project Structure
| File | Purpose |
|------|---------|
| `train.ipynb` | Training notebook for LoRA adapter |
| `merge_lora.py` | Utility to merge LoRA weights |
| `streaming.py` | FastAPI server for real-time inference |
| `streaming_client.py` | Terminal client for testing audio streaming |
| `run_inference.py` | Script for offline batch audio generation |
| `eval_data_25.csv` | Test sentences for evaluation |

> [!NOTE]
> For a detailed breakdown of all data files, preprocessing scripts, and specific audio generation tools, please refer to [DETAILS.md](./DETAILS.md).

---

## Models & Datasets

- **Base Model**: [maya-research/veena-tts](https://huggingface.co/maya-research/veena-tts)
- **Fine-tuned Model**: [akh99/veena-hinglish-stage1](https://huggingface.co/akh99/veena-hinglish-stage1) ⭐ (Best)
- **Training Dataset**: [akh99/indictts-hinglish](https://huggingface.co/datasets/akh99/indictts-hinglish)

---

---

## Technical Details

For in-depth information on:
*   **Data Sources**: (Generated Hinglish vs Indic TTS)
*   **Speaker Management**: (Preserving "kavya" voice)
*   **Model Comparisons**: (Performance benchmarks)
*   **Dataset Processing**: (Hinglish conversion workflows)

👉 **Please see [DETAILS.md](./DETAILS.md)**

---

## Acknowledgments

- Base model: [Maya Research - Veena TTS](https://github.com/maya-research/veena)
- Audio codec: [SNAC](https://github.com/hubertsiuzdak/snac)
- Inference: [vLLM](https://vllm.ai/)
