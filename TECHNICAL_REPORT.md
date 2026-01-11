# Technical Report: Veena Hinglish TTS

**Date**: January 2026
**Model Version**: `veena-hinglish-stage1` (Best)

## 1. Executive Summary

Veena Hinglish TTS is a state-of-the-art text-to-speech model designed specifically for Hindi-English code-switched (Hinglish) speech. Built on top of **Llama 3.2 3B**, the model leverages a speech-language model (SLM) architecture to generate high-fidelity, natural-sounding audio.

Through fine-tuning on a curated dataset of authentic Hindi speech and generated Hinglish content, we achieved a significant quality improvement, raising the Mean Opinion Score (MOS) from **4.12 (Base)** to **4.66 (Fine-tuned)**.

## 2. Architecture

Veena treats speech synthesis as a language modeling task.

*   **Backbone**: **Llama 3.2 3B** (Fine-tuned).
*   **Audio Tokenizer**: **SNAC** (24kHz) at 32kbps.
*   **Methodology**: The model autoregressively predicts audio tokens based on input text tokens.
    *   **Input**: Text tokens + Speaker Embeddings (`<spk_kavya>`).
    *   **Output**: Discrete audio codes corresponding to SNAC layers.

## 3. Training Details

The best performing model, `veena-hinglish-stage1`, was trained using parameter-efficient fine-tuning (LoRA) to adapt the Llama 3.2 backbone to Hinglish prosody and pronunciation.

### Hardware & Performance
*   **GPU Used**: NVIDIA **A100 80GB**
*   **Training Time**: ~1 hour for the best model
*   **Precision**: `bfloat16` (BF16) throughout training

### Hyperparameters & Optimizations
We utilized **LoRA (Low-Rank Adaptation)** to train efficiently with minimal VRAM usage.

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| **LoRA Rank (Attention)** | `192` | High rank for capturing complex prosody |
| **LoRA Rank (FFN)** | `96` | Separate rank for Feed-Forward Networks |
| **LoRA Alpha** | `2x Rank` | 384 (Attn) / 192 (FFN) |
| **Target Modules** | All Linear | `q`, `k`, `v`, `o`, `gate`, `up`, `down` |
| **Optimizer** | `adamw_8bit` | Memory efficency |
| **Learning Rate** | `1e-4` | Cosine scheduler with 2% warmup |
| **Batch Size** | 24 | Per device micro-batch 6 * Grad Accum 4 |
| **Dropout** | 0.05 | To prevent overfitting on small datasets |

## 4. Inference & Real-time Streaming

Inference is optimized for both offline batch generation and low-latency real-time streaming.

### Streaming Architecture
*   **Hardware**: NVIDIA **H100** (Hopper) used for streaming benchmarks.
*   **Engine**: **vLLM** (AsyncLLMEngine).
*   **Precision**: `bfloat16`.
*   **Logic**:
    *   **Sliding Window**: Uses a 7-token stride with a window of 28 tokens (4 frames) to maintain audio coherence across chunks.
    *   **Latency**: First chunk emitted after just 7 tokens generated.

### Optimizations
*   **vLLM Integration**: High-throughput serving layout.
*   **Mixed Precision**: BF16 for faster computation without quality loss.
*   **SNAC Decoding**: Efficient 24kHz audio reconstruction from discrete codes.

## 5. Evaluation Results

We conducted subjective Mean Opinion Score (MOS) testing to validate the model's improvements over the base version.

| Model | MOS Score | Relative Improvement |
| :--- | :--- | :--- |
| **Base Veena Model** | 4.12 / 5 | - |
| **Veena Hinglish (Fine-tuned)** | **4.66 / 5** | **+13%** |

**Key Observations**:
*   The fine-tuned model significantly reduces robotic artifacts in code-mixed sentences.
*   Pronunciation of English words within Hindi sentences ("Office", "Time", "Market") is more natural.
