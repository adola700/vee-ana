"""
FastAPI TTS Server with Real-time Streaming using vLLM
Based on orpheus-streaming architecture
- vLLM for fast token generation
- WebSocket for real-time audio streaming
- SNAC decoder for audio
"""

import os
# Force vLLM v0 engine architecture
os.environ["VLLM_USE_V1"] = "0"
import time
import logging
import asyncio
from typing import AsyncGenerator, List, Optional
from contextlib import asynccontextmanager
import queue
from threading import Thread

import torch
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from starlette.websockets import WebSocketState
from pydantic import BaseModel
from snac import SNAC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Control token IDs
START_OF_SPEECH_TOKEN = 128257
END_OF_SPEECH_TOKEN = 128258
START_OF_HUMAN_TOKEN = 128259
END_OF_HUMAN_TOKEN = 128260
START_OF_AI_TOKEN = 128261
END_OF_AI_TOKEN = 128262
AUDIO_CODE_BASE_OFFSET = 128266

# Orpheus-style decoder settings
MIN_FRAMES_FIRST = 7
MIN_FRAMES_SUBSEQ = 28
PROCESS_EVERY = 7

# Default speaker
DEFAULT_SPEAKER = "kavya"

# Globals
llm = None  # vLLM engine
sampling_params = None
snac_model = None
snac_device = None
tokenizer = None


class TTSRequest(BaseModel):
    text: str
    temperature: float = 0.4
    top_p: float = 0.9


def turn_token_into_id(token_id: int, index: int):
    """Convert token ID to audio code following orpheus pattern."""
    mod = index % 7
    expected_offset = AUDIO_CODE_BASE_OFFSET + (mod * 4096)
    
    if AUDIO_CODE_BASE_OFFSET <= token_id < (AUDIO_CODE_BASE_OFFSET + 7 * 4096):
        audio_code = token_id - expected_offset
        if 0 <= audio_code < 4096:
            return audio_code
    return None


def convert_to_audio(multiframe: List[int], is_first_chunk: bool = False) -> bytes:
    """Convert tokens to PCM audio bytes using SNAC decoder."""
    global snac_model, snac_device
    
    if len(multiframe) < 7:
        return None
    
    num_frames = len(multiframe) // 7
    
    codes_0 = torch.empty((1, num_frames), dtype=torch.int32, device=snac_device)
    codes_1 = torch.empty((1, num_frames * 2), dtype=torch.int32, device=snac_device)
    codes_2 = torch.empty((1, num_frames * 4), dtype=torch.int32, device=snac_device)
    
    for i in range(num_frames):
        base_idx = i * 7
        codes_0[0, i] = multiframe[base_idx]
        codes_1[0, i*2] = multiframe[base_idx + 1]
        codes_1[0, i*2 + 1] = multiframe[base_idx + 4]
        codes_2[0, i*4] = multiframe[base_idx + 2]
        codes_2[0, i*4 + 1] = multiframe[base_idx + 3]
        codes_2[0, i*4 + 2] = multiframe[base_idx + 5]
        codes_2[0, i*4 + 3] = multiframe[base_idx + 6]
    
    if (torch.any(codes_0 < 0) or torch.any(codes_0 > 4096) or
        torch.any(codes_1 < 0) or torch.any(codes_1 > 4096) or
        torch.any(codes_2 < 0) or torch.any(codes_2 > 4096)):
        return None
    
    codes = [codes_0, codes_1, codes_2]
    
    with torch.inference_mode():
        audio_hat = snac_model.decode(codes)
        
        if is_first_chunk:
            audio_slice = audio_hat.squeeze()
        else:
            # Each SNAC frame is 512 samples. 
            # We return only the latest frame (7 tokens worth).
            audio_slice = audio_hat[:, :, -512:].squeeze()
        
        if snac_device == "cuda":
            audio_int16 = (audio_slice * 32767.0).round().to(torch.int16)
            return audio_int16.cpu().numpy().tobytes()
        else:
            audio_np = audio_slice.cpu().numpy()
            return (audio_np * 32767.0).round().astype(np.int16).tobytes()


async def generate_tokens_vllm(text: str, temperature: float, top_p: float):
    """
    Generate tokens using vLLM AsyncLLMEngine for TRUE streaming.
    Yields tokens as they are generated, one by one.
    """
    global llm, tokenizer
    from vllm import SamplingParams
    import uuid
    
    prompt = f"<spk_{DEFAULT_SPEAKER}> {text}"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    
    input_tokens = [
        START_OF_HUMAN_TOKEN,
        *prompt_tokens,
        END_OF_HUMAN_TOKEN,
        START_OF_AI_TOKEN,
        START_OF_SPEECH_TOKEN
    ]
    
    max_tokens = min(int(len(text) * 1.3) * 7 + 21, 700)
    
    # Create sampling params for this request
    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=1.05,
        stop_token_ids=[END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN]
    )
    
    # Format prompt text
    prompt_text = tokenizer.decode(input_tokens)
    request_id = str(uuid.uuid4())
    
    # Track already yielded tokens
    prev_token_count = 0
    gen_start = time.perf_counter()
    first_token_time = None
    
    # Use async generator from engine with prompt_token_ids directly
    # This avoids any tokenizer.decode/encode issues with special tokens
    async for request_output in llm.generate({"prompt_token_ids": input_tokens}, params, request_id=request_id):
        # Get new tokens since last yield
        token_ids = request_output.outputs[0].token_ids
        new_tokens = token_ids[prev_token_count:]
        
        if first_token_time is None and len(new_tokens) > 0:
            first_token_time = time.perf_counter()
            logger.info(f"First token at: {(first_token_time - gen_start)*1000:.2f} ms, token_id: {new_tokens[0]}, batch size: {len(new_tokens)}")
        
        for token_id in new_tokens:
            if prev_token_count < 10:
                logger.info(f"Token {prev_token_count}: {token_id}")
            yield token_id
            prev_token_count += 1


async def tokens_decoder(token_gen) -> AsyncGenerator[bytes, None]:
    """Decode tokens into audio chunks."""
    buffer = []
    count = 0
    first_chunk_sent = False
    decode_start = time.perf_counter()
    
    # Minimal threshold for fastest start (7 tokens = 1 frame = ~21ms audio)
    INITIAL_THRESHOLD = 7 
    
    async for token_id in token_gen:
        audio_code = turn_token_into_id(token_id, count)
        if audio_code is None or audio_code < 0:
            continue
        
        buffer.append(audio_code)
        count += 1
        
        if not first_chunk_sent and count >= INITIAL_THRESHOLD:
            snac_start = time.perf_counter()
            audio = convert_to_audio(buffer, is_first_chunk=True)
            snac_time = (time.perf_counter() - snac_start) * 1000
            if audio is not None:
                first_chunk_sent = True
                total_time = (time.perf_counter() - decode_start) * 1000
                logger.info(f"First audio chunk: {count} tokens collected in {total_time:.2f}ms, SNAC decode: {snac_time:.2f}ms")
                yield audio
        elif first_chunk_sent and count % PROCESS_EVERY == 0:
            # Pass a larger window for SNAC overlap but return only newest samples
            window = buffer[-28:] if len(buffer) >= 28 else buffer
            audio = convert_to_audio(window, is_first_chunk=False)
            if audio is not None:
                yield audio


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup."""
    global llm, snac_model, snac_device, tokenizer
    
    model_name = os.environ.get("MODEL_NAME", "maya-research/veena-tts")
    
    logger.info("Loading SNAC model...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
    snac_device = "cuda" if torch.cuda.is_available() else "cpu"
    snac_model = snac_model.to(snac_device)
    
    if snac_device == "cuda":
        torch.backends.cudnn.benchmark = True
        dummy_codes = [
            torch.randint(0, 4096, (1, 1), dtype=torch.int32, device=snac_device),
            torch.randint(0, 4096, (1, 2), dtype=torch.int32, device=snac_device),
            torch.randint(0, 4096, (1, 4), dtype=torch.int32, device=snac_device)
        ]
        with torch.inference_mode():
            _ = snac_model.decode(dummy_codes)
    
    logger.info(f"Loading vLLM AsyncLLMEngine with model: {model_name}")
    from vllm import AsyncLLMEngine, AsyncEngineArgs
    from transformers import AutoTokenizer
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Create engine args for async engine
    engine_args = AsyncEngineArgs(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        enforce_eager=True,
    )
    
    # Load AsyncLLMEngine for true streaming
    llm = AsyncLLMEngine.from_engine_args(engine_args)
    
    logger.info("Models loaded! Ready to serve.")
    yield
    logger.info("Shutting down...")
    # Shutdown engine
    if hasattr(llm, "shutdown_background_loop"):
        llm.shutdown_background_loop()
    elif hasattr(llm, "shutdown"):
        llm.shutdown()


app = FastAPI(
    title="TTS Streaming Server (vLLM)",
    description="Real-time streaming TTS with vLLM acceleration",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/v1/audio/speech/stream/ws")
async def tts_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time audio streaming."""
    await websocket.accept()
    logger.info("WebSocket connection opened")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if not data.get("continue", True):
                break
            
            text = data.get("text", "").strip()
            if not text:
                await websocket.send_json({"type": "error", "message": "Empty text"})
                continue
            
            temperature = data.get("temperature", 0.4)
            top_p = data.get("top_p", 0.9)
            segment_id = data.get("segment_id", "default")
            
            start_time = time.perf_counter()
            
            try:
                await websocket.send_json({"type": "start", "segment_id": segment_id})
                
                logger.info(f"Generating audio for: '{text[:50]}...'")
                
                token_gen = generate_tokens_vllm(text, temperature, top_p)
                audio_gen = tokens_decoder(token_gen)
                
                chunk_count = 0
                total_bytes = 0
                async for chunk in audio_gen:
                    chunk_count += 1
                    total_bytes += len(chunk)
                    elapsed = (time.perf_counter() - start_time) * 1000
                    
                    if chunk_count == 1:
                        logger.info(f"TTFB: {elapsed:.2f} ms, first chunk size: {len(chunk)} bytes")
                    elif chunk_count <= 3:
                        logger.info(f"Chunk {chunk_count}: {elapsed:.2f} ms, size: {len(chunk)} bytes")
                    
                    await websocket.send_bytes(chunk)
                
                total_time = (time.perf_counter() - start_time) * 1000
                logger.info(f"Generation complete: {chunk_count} chunks, {total_bytes} total bytes, {total_time:.2f} ms total")
                
                await websocket.send_json({"type": "end", "segment_id": segment_id})
                
            except Exception as e:
                logger.exception(f"Error during generation: {e}")
                await websocket.send_json({"type": "error", "message": str(e)})
    
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()


@app.post("/v1/audio/speech/stream")
async def tts_stream(request: TTSRequest):
    """HTTP streaming endpoint."""
    
    async def generate():
        start_time = time.perf_counter()
        first_chunk = True
        
        token_gen = generate_tokens_vllm(request.text, request.temperature, request.top_p)
        audio_gen = tokens_decoder(token_gen)
        
        async for chunk in audio_gen:
            if first_chunk:
                ttfb = time.perf_counter() - start_time
                logger.info(f"TTFB: {ttfb*1000:.2f} ms")
                first_chunk = False
            yield chunk
    
    return StreamingResponse(
        generate(),
        media_type="audio/pcm",
        headers={
            "X-Audio-Sample-Rate": "24000",
            "X-Audio-Channels": "1",
            "X-Audio-Format": "int16"
        }
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the streaming TTS demo page."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>TTS Streaming Demo (vLLM)</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: rgba(255,255,255,0.95);
            border-radius: 16px;
            padding: 40px;
            width: 100%;
            max-width: 600px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 { color: #1a1a2e; margin-bottom: 8px; font-size: 28px; }
        .subtitle { color: #666; margin-bottom: 24px; }
        .badge { background: #10b981; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; }
        .input-group { margin-bottom: 20px; }
        label { display: block; color: #333; font-weight: 600; margin-bottom: 8px; }
        input[type="text"] {
            width: 100%;
            padding: 14px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.2s;
        }
        input[type="text"]:focus { outline: none; border-color: #4a6cf7; }
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #4a6cf7 0%, #6366f1 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(74,108,247,0.4); }
        button:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .status {
            margin-top: 20px;
            padding: 12px 16px;
            border-radius: 8px;
            text-align: center;
            font-weight: 500;
        }
        .status.info { background: #e0f2fe; color: #0277bd; }
        .status.success { background: #e8f5e9; color: #2e7d32; }
        .status.error { background: #ffebee; color: #c62828; }
        .status.generating { background: #fff3e0; color: #ef6c00; }
        .stats { margin-top: 16px; display: flex; gap: 24px; justify-content: center; }
        .stat { text-align: center; }
        .stat-value { font-size: 24px; font-weight: bold; color: #4a6cf7; }
        .stat-label { font-size: 12px; color: #666; margin-top: 4px; }
        .examples { margin-top: 24px; }
        .examples h3 { font-size: 14px; color: #666; margin-bottom: 12px; }
        .example-btn {
            display: inline-block;
            padding: 8px 16px;
            margin: 4px;
            background: #f5f5f5;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .example-btn:hover { background: #4a6cf7; color: white; border-color: #4a6cf7; }
    </style>
</head>
<body>
    <div class="container">
        <h1>TTS Streaming Demo <span class="badge">vLLM</span></h1>
        <p class="subtitle">Real-time Text-to-Speech with Hinglish Support</p>
        
        <div class="input-group">
            <label for="textInput">Enter text and press Enter</label>
            <input type="text" id="textInput" value="Hello! Aaj ka weather kaisa hai?" 
                   placeholder="Type your text here...">
        </div>
        
        <button id="generateBtn">Generate & Play</button>
        
        <div id="status" class="status info">Ready! Type text and press Enter or click Generate</div>
        
        <div class="stats" id="stats" style="display:none;">
            <div class="stat">
                <div class="stat-value" id="ttfb">-</div>
                <div class="stat-label">TTFB (ms)</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="duration">-</div>
                <div class="stat-label">Duration (s)</div>
            </div>
        </div>
        
        <div class="examples">
            <h3>Quick Examples:</h3>
            <span class="example-btn" data-text="Namaste! Aap kaise hain?">Greeting</span>
            <span class="example-btn" data-text="Main office ja raha hoon meeting ke liye.">Office</span>
            <span class="example-btn" data-text="Aaj ka weather bahut accha hai!">Weather</span>
        </div>
    </div>

    <script>
        const textInput = document.getElementById('textInput');
        const generateBtn = document.getElementById('generateBtn');
        const status = document.getElementById('status');
        const stats = document.getElementById('stats');
        const ttfbEl = document.getElementById('ttfb');
        const durationEl = document.getElementById('duration');
        
        let audioContext = null;
        let isGenerating = false;
        
        function initAudio() {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
            }
            if (audioContext.state === 'suspended') {
                audioContext.resume();
            }
        }
        
        function setStatus(msg, type) {
            status.textContent = msg;
            status.className = 'status ' + type;
        }
        
        async function generateSpeech() {
            const text = textInput.value.trim();
            if (!text || isGenerating) return;
            
            isGenerating = true;
            generateBtn.disabled = true;
            setStatus('Connecting...', 'generating');
            stats.style.display = 'none';
            
            initAudio();
            
            let requestSentTime = 0;  // Time when we actually send the request
            let firstChunkTime = 0;
            let totalSamples = 0;
            let nextPlayTime = 0;  // For seamless audio scheduling
            
            try {
                const ws = new WebSocket('ws://' + window.location.host + '/v1/audio/speech/stream/ws');
                
                ws.onopen = () => {
                    setStatus('Generating...', 'generating');
                    // Measure TTFB from when request is actually sent
                    requestSentTime = performance.now();
                    ws.send(JSON.stringify({ text: text, temperature: 0.4, top_p: 0.9 }));
                };
                
                ws.onmessage = async (event) => {
                    if (event.data instanceof Blob) {
                        // First chunk received - record TTFB
                        if (firstChunkTime === 0) {
                            firstChunkTime = performance.now();
                            const ttfb = Math.round(firstChunkTime - requestSentTime);
                            console.log('First audio chunk received at:', ttfb, 'ms');
                            ttfbEl.textContent = ttfb;
                            stats.style.display = 'flex';
                            setStatus('Playing...', 'success');
                            
                            // Initialize playback time to NOW for immediate start
                            nextPlayTime = audioContext.currentTime;
                        }
                        
                        const arrayBuffer = await event.data.arrayBuffer();
                        const int16Array = new Int16Array(arrayBuffer);
                        console.log('Audio chunk size:', int16Array.length, 'samples');
                        
                        // Convert to float32 for Web Audio
                        const float32Array = new Float32Array(int16Array.length);
                        for (let i = 0; i < int16Array.length; i++) {
                            float32Array[i] = int16Array[i] / 32768.0;
                        }
                        
                        // Create and play audio buffer IMMEDIATELY
                        const audioBuffer = audioContext.createBuffer(1, float32Array.length, 24000);
                        audioBuffer.getChannelData(0).set(float32Array);
                        
                        const source = audioContext.createBufferSource();
                        source.buffer = audioBuffer;
                        source.connect(audioContext.destination);
                        
                        // Play at the scheduled time (or now if behind)
                        const playTime = Math.max(nextPlayTime, audioContext.currentTime);
                        console.log('Playing at:', playTime.toFixed(3), 'current:', audioContext.currentTime.toFixed(3));
                        source.start(playTime);
                        nextPlayTime = playTime + audioBuffer.duration;
                        
                        totalSamples += float32Array.length;
                    } else {
                        const data = JSON.parse(event.data);
                        if (data.type === 'end') {
                            durationEl.textContent = (totalSamples / 24000).toFixed(2);
                            setStatus('Complete!', 'success');
                            ws.close();
                        } else if (data.type === 'error') {
                            setStatus('Error: ' + data.message, 'error');
                            ws.close();
                        }
                    }
                };
                
                ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    setStatus('Connection error', 'error');
                };
                
                ws.onclose = () => {
                    isGenerating = false;
                    generateBtn.disabled = false;
                };
                
            } catch (error) {
                console.error('Error:', error);
                setStatus('Error: ' + error.message, 'error');
                isGenerating = false;
                generateBtn.disabled = false;
            }
        }
        
        generateBtn.addEventListener('click', generateSpeech);
        textInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') generateSpeech();
        });
        
        document.querySelectorAll('.example-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                textInput.value = btn.dataset.text;
                generateSpeech();
            });
        });
    </script>
</body>
</html>
"""


@app.get("/health")
async def health():
    return {"status": "healthy", "backend": "vllm", "model_loaded": llm is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
