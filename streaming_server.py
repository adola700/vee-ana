import os
# Force vLLM v0 engine architecture
os.environ["VLLM_USE_V1"] = "0"
import time
import logging
import asyncio
import uuid
import torch
import numpy as np
from typing import AsyncGenerator, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import orjson
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

# Globals
llm = None
tokenizer = None
snac_model = None
snac_device = "cuda" if torch.cuda.is_available() else "cpu"

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
        audio_slice = audio_hat.squeeze() if is_first_chunk else audio_hat[:, :, -512:].squeeze()
        audio_int16 = (audio_slice * 32767.0).clamp(-32768, 32767).round().to(torch.int16)
        return audio_int16.cpu().numpy().tobytes()

async def generate_tokens_vllm(text: str):
    from vllm import SamplingParams
    prompt = f"<spk_{DEFAULT_SPEAKER}> {text}"
    # Standard control tokens for Veena
    input_ids = [128259] + tokenizer.encode(prompt, add_special_tokens=False) + [128260, 128261, 128257]
    params = SamplingParams(temperature=0.4, top_p=0.9, max_tokens=1024, stop_token_ids=[128258, 128262])
    
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
            audio = convert_to_audio(buffer[:7], is_first_chunk=True)
            if audio:
                first_sent = True
                yield audio
                await asyncio.sleep(0) 
        elif first_sent and count % PROCESS_EVERY == 0:
            window = buffer[-MIN_FRAMES_SUBSEQ:]
            audio = convert_to_audio(window, is_first_chunk=False)
            if audio: yield audio

# --- Startup & Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, snac_model, tokenizer
    model_path = "maya-research/veena-tts"
    
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(snac_device)
    if snac_device == "cuda":
        snac_model.decode = torch.compile(snac_model.decode)

    from vllm import AsyncLLMEngine, AsyncEngineArgs
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    args = AsyncEngineArgs(
        model=model_path, trust_remote_code=True, dtype="bfloat16", 
        gpu_memory_utilization=0.4, enforce_eager=True
    )
    llm = AsyncLLMEngine.from_engine_args(args)

    logger.info("Warming up pipeline...")
    async for _ in tokens_decoder(generate_tokens_vllm("Warmup")): break
    yield
    if hasattr(llm, "shutdown_background_loop"):
        llm.shutdown_background_loop()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.websocket("/v1/audio/speech/stream/ws")
async def tts_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            data = orjson.loads(raw)
            text = data.get("input") or data.get("text")
            if not text: continue
            
            await ws.send_text(orjson.dumps({"type": "start"}).decode())
            async for chunk in tokens_decoder(generate_tokens_vllm(text)):
                await ws.send_bytes(chunk)
                # CRITICAL: Prevent chunk batching/buffering
                await asyncio.sleep(0.001)
            await ws.send_text(orjson.dumps({"type": "end"}).decode())
    except WebSocketDisconnect: pass

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Veena H100 Stream</title>
    <style>
        body { font-family: sans-serif; background: #0f172a; color: white; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
        .box { background: white; color: #333; padding: 40px; border-radius: 16px; width: 550px; box-shadow: 0 10px 40px rgba(0,0,0,0.4); }
        textarea { width: 100%; height: 100px; padding: 15px; border-radius: 10px; border: 2px solid #e2e8f0; margin-bottom: 20px; font-size: 16px; outline: none;}
        button { width: 100%; padding: 16px; background: #2563eb; color: white; border: none; border-radius: 10px; font-weight: bold; cursor: pointer; font-size: 18px;}
        #status { margin-top: 25px; text-align: center; color: #64748b; font-weight: 600; }
    </style>
</head>
<body>
    <div class="box">
        <h2 style="text-align:center; color:#0f172a">Veena TTS Engine</h2>
        <textarea id="inp">Namaste! Sample rate and buffering fixed. Testing human speed audio.</textarea>
        <button id="btn">Speak Now</button>
        <div id="status">Ready (24kHz Optimized)</div>
    </div>
    <script>
        const btn = document.getElementById('btn');
        const inp = document.getElementById('inp');
        const status = document.getElementById('status');
        let audioCtx, nextTime = 0;

        btn.onclick = async () => {
            if (!audioCtx) {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)({ 
                    sampleRate: 24000,
                    latencyHint: 'interactive'
                });
            }
            if (audioCtx.state === 'suspended') await audioCtx.resume();
            
            btn.disabled = true;
            status.innerText = "Connecting...";
            nextTime = 0; // Reset timeline

            const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
            const ws = new WebSocket(protocol + window.location.host + '/v1/audio/speech/stream/ws');
            ws.binaryType = "blob";

            ws.onopen = () => ws.send(JSON.stringify({ input: inp.value }));
            
            ws.onmessage = async (e) => {
                if (e.data instanceof Blob) {
                    const arrayBuf = await e.data.arrayBuffer();
                    const pcm = new Int16Array(arrayBuf);
                    const f32 = new Float32Array(pcm.length);
                    for (let i=0; i<pcm.length; i++) f32[i] = pcm[i]/32768.0;
                    
                    const audioBuf = audioCtx.createBuffer(1, f32.length, 24000);
                    audioBuf.getChannelData(0).set(f32);
                    const src = audioCtx.createBufferSource();
                    src.buffer = audioBuf;
                    src.connect(audioCtx.destination);
                    
                    const now = audioCtx.currentTime;
                    // Sequence synchronization logic
                    if (nextTime < now) {
                        nextTime = now + 0.05; // Initial 50ms buffer
                    }
                    
                    src.start(nextTime);
                    nextTime += audioBuf.duration; // Chain exactly at the end
                    status.innerText = "Streaming Audio...";
                } else {
                    const d = JSON.parse(e.data);
                    if (d.type === 'end') {
                        status.innerText = "Done";
                        btn.disabled = false;
                        ws.close();
                    }
                }
            };
        };
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)