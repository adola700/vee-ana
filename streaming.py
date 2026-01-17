import os
os.environ["VLLM_USE_V1"] = "0"
import time, logging, asyncio, uuid, torch
import numpy as np
from typing import AsyncGenerator, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from snac import SNAC
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
AUDIO_CODE_BASE_OFFSET = 128266  
MIN_FRAMES_FIRST = 7*4   # CHANGED: Wait for 4 frames (28 tokens) before starting
PROCESS_EVERY = 7        # CHANGED: Process every 1 frame (7 tokens) thereafter
SAMPLES_PER_FRAME = 2048*2 
SAMPLE_RATE = 24000

llm = None
tokenizer = None
snac_model = None
snac_device = "cuda" if torch.cuda.is_available() else "cpu"

def turn_token_into_id(token_id: int, index: int):
    mod = index % 7
    audio_code = token_id - (AUDIO_CODE_BASE_OFFSET + (mod * 4096))
    # Validate: audio codes should be 0-4095
    if 0 <= audio_code < 4096:
        return audio_code
    else:
        return None  # Invalid code - skip it

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
        
        # --- CHANGED: Sliding Window Slicing Logic ---
        if is_first_chunk:
            # First chunk: Return the first ~2 frames (4096 samples)
            audio_slice = audio_hat[:, :, :4096].squeeze()
        else:
            # Steady state: Always take the "middle" stable slice (samples 2048 to 4096)
            # This corresponds to the 2nd frame in the 4-frame window
            audio_slice = audio_hat[:, :, 2048:4096].squeeze()
        
        return (audio_slice * 32767.0).clamp(-32768, 32767).round().to(torch.int16).cpu().numpy().tobytes()

async def generate_tokens_vllm(text: str):
    from vllm import SamplingParams
    prompt = f"<spk_kavya> {text}"
    input_ids = [128259] + tokenizer.encode(prompt, add_special_tokens=False) + [128260, 128261, 128257]
    params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=1024, stop_token_ids=[128258, 128262])
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
        if code is None: 
            continue
        buffer.append(code)
        count += 1
        
        if not first_sent and count >= MIN_FRAMES_FIRST:
            audio = convert_to_audio(buffer, is_first_chunk=True)
            if audio:
                first_sent = True
                yield audio
                await asyncio.sleep(0.001)
        
        elif first_sent and count % PROCESS_EVERY == 0:
            # Sliding window: Keep last 28 tokens (4 frames) for context
            # This logic works automatically with the new PROCESS_EVERY=7
            window = buffer[-28:] 
            audio = convert_to_audio(window, is_first_chunk=False)
            if audio:
                yield audio
                await asyncio.sleep(0.001)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, snac_model, tokenizer
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(snac_device)
    from vllm import AsyncLLMEngine, AsyncEngineArgs
    tokenizer = AutoTokenizer.from_pretrained("akh99/veena-hinglish-stage1")
    args = AsyncEngineArgs(model="akh99/veena-hinglish-stage1", dtype="bfloat16", gpu_memory_utilization=0.8, enforce_eager=True, max_model_len= 2048)
    llm = AsyncLLMEngine.from_engine_args(args)
    yield

app = FastAPI(lifespan=lifespan)
@app.websocket("/v1/audio/speech/stream/ws")
async def tts_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            text = data.get("text")
            last_chunk_time = time.perf_counter()
            total_samples = 0
            start_time = last_chunk_time

            async for chunk in tokens_decoder(
                generate_tokens_vllm(text)
            ):
                now = time.perf_counter()

                # ---- INSTANTANEOUS RTF ----
                wall_dt = now - last_chunk_time

                samples = len(chunk) // 2
                audio_dt = samples / SAMPLE_RATE

                inst_rtf = (
                    wall_dt / audio_dt
                    if audio_dt > 0 else float("inf")
                )

                logger.info(
                    f"inst_RTF={inst_rtf:.2f} | "
                    f"chunk_audio={audio_dt*1000:.1f}ms | "
                    f"wall={wall_dt*1000:.1f}ms"
                )

                last_chunk_time = now
                total_samples += samples

                await ws.send_bytes(chunk)
                await asyncio.sleep(0.001)

            # ---- AVERAGE RTF ----
            elapsed = time.perf_counter() - start_time
            audio_seconds = total_samples / SAMPLE_RATE
            avg_rtf = elapsed / audio_seconds if audio_seconds > 0 else float("inf")

            logger.info(
                f"AVG_RTF={avg_rtf:.3f} | "
                f"audio={audio_seconds:.2f}s | wall={elapsed:.2f}s"
            )

            await ws.send_json({
                "type": "end",
                "avg_rtf": round(avg_rtf, 3),
            })

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)