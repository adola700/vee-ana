import asyncio
import websockets
import json
import pyaudio
import numpy as np
from collections import deque
import time
import logging
import wave

# DEBUG: Save full audio to file for analysis
DEBUG_SAVE_AUDIO = True


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Install: pip install pyaudio websockets numpy
WS_URL = "ws://localhost:8000/v1/audio/speech/stream/ws"
TARGET_RATE = 24000  # Model output - match this exactly
BUFFER_MS = 300  # Initial buffer before playback starts
JITTER_BUFFER_SIZE = 10  # Keep N chunks ahead in sliding window

async def local_stream():
    p = pyaudio.PyAudio()
    hw_rate = TARGET_RATE
    
    print(f"\nStreaming at {hw_rate}Hz")
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=hw_rate, output=True, frames_per_buffer=1024)
    
    try:
        async with websockets.connect(WS_URL) as ws:
            print("--- H100 Terminal TTS Ready ---\n")
            while True:
                text = input("Enter text to speak: ")
                if not text.strip(): continue
                
                await ws.send(json.dumps({"text": text}))
                print("[SYSTEM]: Starting playback...", end="", flush=True)
                
                audio_queue = deque(maxlen=JITTER_BUFFER_SIZE)
                receiving = True
                buffered_samples = 0
                buffer_ready = asyncio.Event()
                
                # DEBUG: Collect all chunks for saving
                all_audio_chunks = [] if DEBUG_SAVE_AUDIO else None
                
                # Receive all chunks first
                async def receive_chunks():
                    nonlocal receiving, buffered_samples
                    chunk_count = 0
                    last_chunk_time = time.perf_counter()
                    start_time = last_chunk_time
                    total_samples = 0

                    while True:
                        t_recv_start = time.perf_counter()
                        msg = await ws.recv()
                        recv_dt = time.perf_counter() - t_recv_start

                        if isinstance(msg, bytes):
                            audio_data = np.frombuffer(msg, dtype=np.int16)
                            now = time.perf_counter()
                            
                            wall_dt = now - last_chunk_time
                            
                            samples = len(audio_data)
                            audio_dt = samples / TARGET_RATE

                            inst_rtf = (
                                wall_dt / audio_dt
                                if audio_dt > 0 else float("inf")
                            )

                            is_buffered = recv_dt < 0.001
                            buffered_tag = " [BUFFERED]" if is_buffered else ""
                            
                            logger.info(
                                f"inst_RTF={inst_rtf:.2f}{buffered_tag} | "
                                f"chunk_audio={audio_dt*1000:.1f}ms | "
                                f"wall={wall_dt*1000:.1f}ms | "
                                f"recv={recv_dt*1000:.1f}ms"
                            )

                            last_chunk_time = now
                            total_samples += samples

                            audio_queue.append(audio_data)
                            buffered_samples += samples
                            
                            # DEBUG: Store chunk for saving
                            if DEBUG_SAVE_AUDIO:
                                all_audio_chunks.append(audio_data)
                            chunk_count += 1
                            
                            # Signal buffer ready once we have enough
                            if not buffer_ready.is_set() and buffered_samples >= TARGET_RATE * (BUFFER_MS / 1000):
                                logger.info(f"Buffer ready: {buffered_samples} samples ({buffered_samples / TARGET_RATE * 1000:.0f}ms)")
                                buffer_ready.set()
                        else:
                            if json.loads(msg).get("type") == "end":
                                elapsed = time.perf_counter() - start_time
                                audio_seconds = total_samples / TARGET_RATE
                                avg_rtf = elapsed / audio_seconds if audio_seconds > 0 else float("inf")
                                
                                logger.info(
                                    f"AVG_RTF={avg_rtf:.3f} | "
                                    f"audio={audio_seconds:.2f}s | wall={elapsed:.2f}s"
                                )

                                receiving = False
                                break
                
                # Start receiving in background
                receive_task = asyncio.create_task(receive_chunks())
                
                # Wait for initial buffer before starting playback
                print(f"\n[SYSTEM]: Buffering {BUFFER_MS}ms...", end="", flush=True)
                await buffer_ready.wait()
                print(" Ready!")
                
                # Play from jitter buffer (sliding window)
                chunk_idx = 0
                while True:
                    # Wait for chunk in queue
                    while len(audio_queue) == 0 and receiving:
                        await asyncio.sleep(0.01)  # Poll quickly
                    
                    if len(audio_queue) > 0:
                        audio_chunk = audio_queue.popleft()
                        num_samples = len(audio_chunk)
                        duration_sec = num_samples / TARGET_RATE
                        
                        # Play this chunk (non-blocking)
                        await asyncio.get_running_loop().run_in_executor(None, stream.write, audio_chunk.tobytes())
                        chunk_idx += 1
                        print(f"\r[SYSTEM]: Playing chunk {chunk_idx}... (queue: {len(audio_queue)})", end="", flush=True)
                    else:
                        # Queue empty and receiving done
                        if not receiving:
                            break
                
                print("\r[SYSTEM]: Playback complete.")
                await receive_task
                
                # DEBUG: Save concatenated audio to WAV file
                if DEBUG_SAVE_AUDIO and all_audio_chunks:
                    full_audio = np.concatenate(all_audio_chunks)
                    filename = f"debug_audio_{int(time.time())}.wav"
                    with wave.open(filename, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(TARGET_RATE)
                        wf.writeframes(full_audio.tobytes())
                    print(f"[DEBUG]: Saved {len(full_audio)} samples to {filename}")
    
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
    finally:
        stream.close()
        p.terminate()

if __name__ == "__main__":
    asyncio.run(local_stream())