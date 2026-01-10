# TTS Streaming Server

Real-time streaming Text-to-Speech server using vLLM + SNAC, based on [orpheus-streaming](https://github.com/taresh18/orpheus-streaming) architecture.

## Features

- **Real-time WebSocket streaming** - Audio plays as chunks are generated
- **Low latency** - ~160ms TTFB with vLLM AsyncLLMEngine
- **Built-in Web UI** - Modern HTML interface at `/`
- **REST API** - HTTP streaming endpoint
- **Hinglish support** - Works with English-Hindi mixed text
- **FP16 precision** - No quantization for best quality

## Quick Start

### 1. Install Dependencies

```bash
source venv/bin/activate
pip install fastapi uvicorn websockets vllm
```

### 2. Run the Server

```bash
source venv/bin/activate
python streaming_server.py
```

Server starts at **http://localhost:8000**

### 3. Open Web UI

Go to **http://localhost:8000** in your browser, type text and press Enter!

---

## API Endpoints

### WebSocket Streaming (Recommended)

**Endpoint:** `ws://localhost:8000/v1/audio/speech/stream/ws`

This is the primary endpoint for real-time audio streaming.

**Send JSON:**
```json
{
  "text": "Hello! Aaj ka weather kaisa hai?",
  "temperature": 0.4,
  "top_p": 0.9
}
```

**Receive:**
1. `{"type": "start", "segment_id": "..."}` - Generation started
2. Binary audio chunks (PCM int16, 24kHz mono) - Play immediately!
3. `{"type": "end", "segment_id": "..."}` - Generation complete

**JavaScript Example:**
```javascript
const ws = new WebSocket('ws://localhost:8000/v1/audio/speech/stream/ws');

ws.onopen = () => {
    ws.send(JSON.stringify({
        text: "Namaste! Aap kaise hain?",
        temperature: 0.4,
        top_p: 0.9
    }));
};

ws.onmessage = async (event) => {
    if (event.data instanceof Blob) {
        // Binary audio chunk - play it!
        const arrayBuffer = await event.data.arrayBuffer();
        const int16Array = new Int16Array(arrayBuffer);
        // ... play using Web Audio API
    } else {
        const data = JSON.parse(event.data);
        console.log('Message:', data.type);
    }
};
```

---

### HTTP Streaming

**Endpoint:** `POST /v1/audio/speech/stream`

**Request:**
```bash
curl -X POST "http://localhost:8000/v1/audio/speech/stream" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello! Aaj ka weather kaisa hai?", "temperature": 0.4, "top_p": 0.9}' \
  --output speech.pcm
```

**Response:** Streaming PCM audio (int16, 24kHz mono)

**Convert to WAV:**
```bash
ffmpeg -f s16le -ar 24000 -ac 1 -i speech.pcm speech.wav
```

**Python Example:**
```python
import requests

url = "http://localhost:8000/v1/audio/speech/stream"
data = {
    "text": "Main office ja raha hoon meeting ke liye.",
    "temperature": 0.4,
    "top_p": 0.9
}

response = requests.post(url, json=data, stream=True)

# Save streamed audio
with open("output.pcm", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)

# Convert PCM to WAV using pydub or ffmpeg
```

---

### Health Check

**Endpoint:** `GET /health`

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{"status": "healthy", "backend": "vllm", "model_loaded": true}
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `maya-research/veena-tts` | HuggingFace model name |

### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.4 | Sampling temperature |
| `top_p` | 0.9 | Nucleus sampling threshold |

---

## Audio Format

- **Sample Rate:** 24,000 Hz
- **Channels:** 1 (Mono)
- **Format:** PCM signed 16-bit little-endian

---

## Architecture

Based on orpheus-streaming's token decoding pattern with vLLM:

1. **Token Generation** - vLLM AsyncLLMEngine generates tokens with true streaming
2. **Token Decoding** - Convert tokens to audio codes using 7-token frames
3. **SNAC Decoding** - Decode audio codes to PCM audio
4. **Streaming** - Send audio chunks via WebSocket as they're generated

```
Text -> vLLM -> Tokens -> SNAC Decoder -> PCM Audio -> WebSocket -> Browser
                |
         First chunk after 7 tokens (~160ms TTFB)
         Subsequent chunks every 7 tokens
```

---

## Files

- `streaming_server.py` - FastAPI server with vLLM + WebSocket streaming
- `run_inference.py` - Batch inference for generating audio files

---

## Troubleshooting

### Port already in use
Change port in `streaming_server.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8001)
```

### CUDA out of memory
Reduce `gpu_memory_utilization` in `streaming_server.py`:
```python
gpu_memory_utilization=0.3
```

### Audio sounds choppy
This can happen if the browser can't keep up with audio scheduling. Try a faster network connection.

---

## Credits

Token decoding pattern based on [orpheus-streaming](https://github.com/taresh18/orpheus-streaming)
