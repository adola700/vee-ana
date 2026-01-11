import asyncio
import websockets
import json
import pyaudio
import numpy as np
from collections import deque
import time

# Install: pip install pyaudio websockets numpy
WS_URL = "ws://localhost:8000/v1/audio/speech/stream/ws"
TARGET_RATE = 24000  # Model output - match this exactly
BUFFER_THRESHOLD = 8

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
                
                chunks = []
                receiving = True
                
                # Receive all chunks first
                async def receive_chunks():
                    nonlocal receiving
                    chunk_count = 0
                    while True:
                        msg = await ws.recv()
                        if isinstance(msg, bytes):
                            audio_data = np.frombuffer(msg, dtype=np.int16)
                            chunks.append(audio_data)
                            chunk_count += 1
                            print(f"\r[SYSTEM]: Received {chunk_count} chunks...", end="", flush=True)
                        else:
                            if json.loads(msg).get("type") == "end":
                                receiving = False
                                break
                
                # Start receiving in background
                receive_task = asyncio.create_task(receive_chunks())
                
                # Play chunks as they arrive, one after another
                chunk_idx = 0
                while True:
                    # Wait for next chunk to be available
                    while chunk_idx >= len(chunks) and receiving:
                        await asyncio.sleep(0.05)  # Wait 50ms for next chunk
                    
                    if chunk_idx < len(chunks):
                        audio_chunk = chunks[chunk_idx]
                        num_samples = len(audio_chunk)
                        duration_sec = num_samples / TARGET_RATE
                        
                        # Play this chunk
                        stream.write(audio_chunk.tobytes())
                        print(f"\r[SYSTEM]: Playing chunk {chunk_idx + 1}... ({duration_sec:.2f}s)", end="", flush=True)
                        chunk_idx += 1
                    else:
                        # No more chunks and receiving done
                        if not receiving:
                            break
                
                print("\r[SYSTEM]: Playback complete.                    ")
                await receive_task
    
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
    finally:
        stream.close()
        p.terminate()

if __name__ == "__main__":
    asyncio.run(local_stream())
