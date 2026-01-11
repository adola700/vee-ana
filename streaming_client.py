import asyncio
import websockets
import json
import pyaudio

# Install: pip install pyaudio websockets
WS_URL = "ws://localhost:8000/v1/audio/speech/stream/ws"

async def local_stream():
    p = pyaudio.PyAudio()
    # 24kHz matches the SNAC model output exactly
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    
    try:
        async with websockets.connect(WS_URL) as ws:
            print("\n--- H100 Terminal TTS Ready ---")
            while True:
                text = input("\nEnter text to speak: ")
                if not text.strip(): continue
                
                await ws.send(json.dumps({"text": text}))
                print("Streaming...", end="", flush=True)
                
                while True:
                    msg = await ws.recv()
                    if isinstance(msg, bytes):
                        stream.write(msg) # Immediate playback
                    else:
                        if json.loads(msg).get("type") == "end":
                            print(" Done.")
                            break
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        stream.close()
        p.terminate()

if __name__ == "__main__":
    asyncio.run(local_stream())