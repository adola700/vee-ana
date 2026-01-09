import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from IPython.display import Audio

load_dotenv()

def generate_openai_speech(text, model="gpt-4o-mini-tts", voice="alloy", output_file="openai_speech.mp3"):
    """
    Generates speech using OpenAI API and returns an IPython Audio object for playback.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    output_path = Path(output_file)
    
    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text
    ) as response:
        response.stream_to_file(output_path)
        
    return Audio(output_file)
