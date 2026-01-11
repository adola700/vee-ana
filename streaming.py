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
from snac import SNAC
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration & Offsets ---
AUDIO_CODE_BASE_OFFSET = 128266  
MIN_FRAMES_FIRST = 7     # Wait for 1 frame (7 tokens) before starting
PROCESS_EVERY = 7        # Process every 1 frame (stride)
CONTEXT_WINDOW = 28      # Keep last 4 frames (28 tokens) for context

# Globals
llm = None
tokenizer = None
snac_model = None
snac_device = "cuda" if torch.cuda.is_available() else "cpu"

def turn_token_into_id(token_id: int, index: int):
    mod = index % 7
    expected_offset = AUDIO_CODE_BASE_OFFSET + (mod * 4096)
    audio_code = token_id - expected_offset
    return audio_code if 0 <= audio_code < 4096 else None

def convert_to_audio(multiframe: List[int], is_first_chunk: bool = False) -> bytes:
    # 1. Determine dimensions
    num_frames = len(multiframe) // 7
    if num_frames == 0: return None

    # 2. Prepare Tensors
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
        
        # --- THE FIX: DYNAMIC SLICING ---
        total_samples = audio_hat.shape[-1]
        
        if is_first_chunk:
            # Send everything we have
            audio_slice = audio_hat.squeeze()
            logger.info(f">> First Chunk: {total_samples} samples")
        else:
            # We stepped forward by 1 frame (PROCESS_EVERY=7).
            # We must output exactly 1 frame worth of audio.
            # Calculate samples per frame (e.g., 8192 / 4 = 2048)
            samples_per_frame = total_samples // num_frames
            
            # Slice the NEWEST samples
            audio_slice = audio_hat[:, :, -samples_per_frame:].squeeze()
            
            # Debug log to confirm fix
            # logger.info(f">> Subseq Chunk: {audio_slice.shape[0]} samples (Calculated slice)")

        # Convert to PCM16
        audio_int16 = (audio_slice * 32767.0).clamp(-32768, 32767).round().to(torch.int16)
        return audio_int16.cpu().numpy().tobytes()

async def generate_tokens_vllm(text: str):
    from vllm import SamplingParams
    prompt = f"<spk_kavya> {text}"
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
        
        # Trigger on first frame
        if not first_sent and count >= MIN_FRAMES_FIRST:
            audio = convert_to_audio(buffer, is_first_chunk=True)
            if audio:
                first_sent = True
                yield audio
                await asyncio.sleep(0.001) # Force Flush
        
        # Trigger on subsequent frames (every 7 tokens)
        elif first_sent and count % PROCESS_EVERY == 0:
            window = buffer[-CONTEXT_WINDOW:] # Use sliding window for context
            audio = convert_to_audio(window, is_first_chunk=False)
            if audio:
                yield audio
                await asyncio.sleep(0.001) # Force Flush

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, snac_model, tokenizer
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(snac_device)
    from vllm import AsyncLLMEngine, AsyncEngineArgs
    tokenizer = AutoTokenizer.from_pretrained("maya-research/veena-tts")
    args = AsyncEngineArgs(
        model="maya-research/veena-tts", 
        dtype="bfloat16", 
        gpu_memory_utilization=0.4, 
        enforce_eager=True
    )
    llm = AsyncLLMEngine.from_engine_args(args)
    yield

app = FastAPI(lifespan=lifespan)

@app.websocket("/v1/audio/speech/stream/ws")
async def tts_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            async for chunk in tokens_decoder(generate_tokens_vllm(data.get("text"))):
                await ws.send_bytes(chunk)
                # CRITICAL: Sleep allows the loop to send the packet immediately
                await asyncio.sleep(0.001)
            await ws.send_json({"type": "end"})
    except WebSocketDisconnect: pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)