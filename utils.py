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

def hindi_to_hinglish(hindi_text: str, model: str = "gpt-4o-mini") -> str:
    """
    Converts Hindi text to Hinglish using OpenAI API.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = f"""
You are an expert at Romanizing Hindi into Hinglish (Hindi written in English letters).

Rules:
- Do NOT translate to English
- Use only English (Roman) letters
- Preserve pronunciation, meaning, and natural spoken Hindi
- Output only the Hinglish text

Examples:

Hindi: मुझे आज ऑफिस जाना है
Hinglish: Mujhe aaj office jaana hai

Hindi: क्या तुम कल आओगे?
Hinglish: Kya tum kal aaoge?

Hindi: भारत एक महान देश है
Hinglish: Bharat ek mahaan desh hai

Now convert:

Hindi: {hindi_text}
Hinglish:
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=600
    )

    return response.choices[0].message.content.strip()
