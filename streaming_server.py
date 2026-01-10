import os
# Force vLLM v0 engine architecture
os.environ["VLLM_USE_V1"] = "0"
import time
import logging
import asyncio
import json
import uuid
import torch
import numpy as np
from typing import AsyncGenerator, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from snac import SNAC
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration & Offsets ---
AUDIO_CODE_BASE_OFFSET = 128266  
MIN_FRAMES_FIRST = 7   
MIN_FRAMES_SUBSEQ = 28 
PROCESS_EVERY = 7      
DEFAULT_SPEAKER = "kavya"

# Special Control Tokens
START_OF_HUMAN_TOKEN = 128259
END_OF_HUMAN_TOKEN = 128260
START_OF_AI_TOKEN = 128261
START_OF_SPEECH_TOKEN = 128257
END_OF_SPEECH_TOKEN = 128258
END_OF_AI_TOKEN = 128262

# Globals
llm = None
tokenizer = None
snac_model = None
snac_device = "cuda" if torch.cuda.is_available() else "cpu"

class TTSRequest(BaseModel):
    text: str
    temperature: float = 0.4
    top_p: float = 0.9

# --- Core Logic Functions ---

def turn_token_into_id(token_id: int, index: int):
    mod = index % 7
    expected_offset = AUDIO_CODE_BASE_OFFSET + (mod * 4096)
    audio_code = token_id - expected_offset
    return audio_code if 0 <= audio_code < 4096 else None

def convert_to_audio(multiframe: List[int], is_first_chunk: bool = False) -> bytes:
    num_frames = len(multiframe) // 7
    if num_frames == 0: return None

    codes_0 = torch.empty((1, num_frames), dtype=torch.int32, device=snac_device)
    codes_1 = torch.empty((1, num_frames * 2), dtype=torch.int32, device=snac_device)
    codes_2 = torch.empty((1, num_frames * 4), dtype=torch.int32, device=snac_device)
    
    for i in range(num_frames):
        b = i * 7
        codes_0[0, i] = multiframe[b]
        codes_1[0, i*2], codes_1[0, i*2 + 1] = multiframe[b+1], multiframe[b+4]
        codes_2[0, i*4:i*4+4] = torch.tensor([multiframe[b+2], multiframe[b+3], multiframe[b+5], multiframe[b+6]], device=snac_device)

    with torch.inference_mode():
        audio_hat = snac_model.decode([codes_0, codes_1, codes_2])
        if is_first_chunk:
            audio_slice = audio_hat.squeeze()
        else:
            audio_slice = audio_hat[:, :, -512:].squeeze()

        audio_int16 = (audio_slice * 32767.0).clamp(-32768, 32767).round().to(torch.int16)
        return audio_int16.cpu().numpy().tobytes()

async def generate_tokens_vllm(text: str, temperature: float, top_p: float):
    from vllm import SamplingParams
    prompt = f"<spk_{DEFAULT_SPEAKER}> {text}"
    input_ids = [START_OF_HUMAN_TOKEN] + tokenizer.encode(prompt, add_special_tokens=False) + \
                [END_OF_HUMAN_TOKEN, START_OF_AI_TOKEN, START_OF_SPEECH_TOKEN]
    
    params = SamplingParams(
        temperature=temperature, top_p=top_p, 
        max_tokens=700, repetition_penalty=1.05,
        stop_token_ids=[END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN]
    )
    
    request_id = str(uuid.uuid4())
    prev_count = 0
    async for output in llm.generate({"prompt_token_ids": input_ids}, params, request_id):
        new_tokens = output.outputs[0].token_ids[prev_count:]
        for t in new_tokens:
            yield t
            prev_count += 1

async def tokens_decoder(token_gen) -> AsyncGenerator[bytes, None]:
    buffer, count, first_sent = [], 0, False
    async for token_id in token_gen:
        code = turn_token_into_id(token_id, count)
        if code is None: continue
        buffer.append(code)
        count += 1

        if not first_sent and count >= MIN_FRAMES_FIRST:
            audio = convert_to_audio(buffer, is_first_chunk=True)
            if audio: first_sent = True; yield audio
        elif first_sent and count % PROCESS_EVERY == 0:
            window = buffer[-MIN_FRAMES_SUBSEQ:]
            audio = convert_to_audio(window, is_first_chunk=False)
            if audio: yield audio

# --- Robust Lifespan & Warmup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, snac_model, tokenizer
    model_path = os.environ.get("MODEL_NAME", "maya-research/veena-tts")
    
    # 1. Load SNAC
    logger.info("Loading SNAC...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(snac_device)
    if snac_device == "cuda":
        snac_model.decode = torch.compile(snac_model.decode)
        logger.info("SNAC Compiled.")

    # 2. Load vLLM
    from vllm import AsyncLLMEngine, AsyncEngineArgs
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    args = AsyncEngineArgs(model=model_path, trust_remote_code=True, dtype="float16", 
                           gpu_memory_utilization=0.4, enforce_eager=True)
    llm = AsyncLLMEngine.from_engine_args(args)

    # 3. FULL PIPELINE WARMUP (The Fix)
    # This runs a real text through the engine and decoder to force CUDA kernels to load
    logger.info("WARMING UP ENGINE: Simulating first speech request...")
    try:
        warmup_text = "Checking system readiness."
        token_gen = generate_tokens_vllm(warmup_text, 0.4, 0.9)
        async for _ in tokens_decoder(token_gen):
            pass 
        
        if snac_device == "cuda":
            torch.cuda.synchronize()
        logger.info("SYSTEM READY: All models primed.")
    except Exception as e:
        logger.error(f"Warmup failed: {e}")

    yield
    if hasattr(llm, "shutdown_background_loop"):
        llm.shutdown_background_loop()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Endpoints ---

@app.websocket("/v1/audio/speech/stream/ws")
async def tts_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            text = data.get("text", "").strip()
            if not text: continue
            
            await websocket.send_json({"type": "start"})
            token_gen = generate_tokens_vllm(text, data.get("temperature", 0.4), data.get("top_p", 0.9))
            async for chunk in tokens_decoder(token_gen):
                await websocket.send_bytes(chunk)
            await websocket.send_json({"type": "end"})
    except WebSocketDisconnect: logger.info("Client left")

@app.get("/", response_class=HTMLResponse)
async def index():
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
            
            let requestSentTime = 0;  
            let firstChunkTime = 0;
            let totalSamples = 0;
            let nextPlayTime = 0;  
            
            try {
                const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
                const ws = new WebSocket(protocol + window.location.host + '/v1/audio/speech/stream/ws');
                
                ws.onopen = () => {
                    setStatus('Generating...', 'generating');
                    requestSentTime = performance.now();
                    ws.send(JSON.stringify({ text: text, temperature: 0.4, top_p: 0.9 }));
                };
                
                ws.onmessage = async (event) => {
                    if (event.data instanceof Blob) {
                        if (firstChunkTime === 0) {
                            firstChunkTime = performance.now();
                            const ttfb = Math.round(firstChunkTime - requestSentTime);
                            ttfbEl.textContent = ttfb;
                            stats.style.display = 'flex';
                            setStatus('Playing...', 'success');
                            nextPlayTime = audioContext.currentTime;
                        }
                        
                        const arrayBuffer = await event.data.arrayBuffer();
                        const int16Array = new Int16Array(arrayBuffer);
                        
                        const float32Array = new Float32Array(int16Array.length);
                        for (let i = 0; i < int16Array.length; i++) {
                            float32Array[i] = int16Array[i] / 32768.0;
                        }
                        
                        const audioBuffer = audioContext.createBuffer(1, float32Array.length, 24000);
                        audioBuffer.getChannelData(0).set(float32Array);
                        
                        const source = audioContext.createBufferSource();
                        source.buffer = audioBuffer;
                        source.connect(audioContext.destination);
                        
                        const playTime = Math.max(nextPlayTime, audioContext.currentTime);
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)