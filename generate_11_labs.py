"""
Hinglish Text-to-Speech Generator using ElevenLabs API

USAGE:
    python generate_audio.py                    # Generate NUM_SAMPLES samples
    python generate_audio.py 50                 # Generate 50 samples
    python generate_audio.py 0 100              # Generate samples from index 0 to 100
    python generate_audio.py 100 200            # Generate samples from index 100 to 200

BEST VOICES FOR HINGLISH:
- Hope (tnSpp4vdxKPjI9w0GnoV) - Upbeat and Clear - RECOMMENDED for Hinglish
- Devi (MF4J4IDTRo0AxOO4dpFR) - Encouraging, Motivating - RECOMMENDED for Hindi
- Aria (9BWtsMINqrJLrRacOk9x) - Expressive female

MODELS (all support Hindi):
- eleven_v3: Latest model (alpha), 74 languages, best quality -- Using this
- eleven_multilingual_v2: Stable, 29 languages, proven quality
- eleven_turbo_v2_5: Faster, 32 languages
"""

import requests
import sys
from pathlib import Path
import time

# ============================================================================
# CONFIGURATION - Modify these as needed
# ============================================================================

API_KEY = "sk_fd314ccaab680e5b8d3efedef406fb91a827cd7c49f99e88"
BASE_URL = "https://api.elevenlabs.io/v1"

# Model options (all support Hindi)
MODEL_ID = "eleven_v3"  # Options: eleven_v3, eleven_multilingual_v2, eleven_turbo_v2_5

# Available voices with Hindi support
VOICES = {
    # RECOMMENDED FOR HINGLISH
    "devi": {
        "id": "MF4J4IDTRo0AxOO4dpFR",
        "desc": "Encouraging and Motivating - BEST for Hindi",
        "gender": "female"
    },
    # Other good options
    "aria": {
        "id": "9BWtsMINqrJLrRacOk9x",
        "desc": "Expressive, natural female",
        "gender": "female"
    },
    "roger": {
        "id": "CwhRBWXzGAHq8TQ4Fs17",
        "desc": "Laid-Back, Casual, Resonant",
        "gender": "male"
    },
    "sarah": {
        "id": "EXAVITQu4vr4xnSDxMaL",
        "desc": "Mature, Reassuring, Confident",
        "gender": "female"
    },
    "george": {
        "id": "JBFqnCBsd6RMkjVDRZzb",
        "desc": "Warm, Captivating Storyteller",
        "gender": "male"
    },
    "charlie": {
        "id": "IKne3meq5aSn9XLyUdCD",
        "desc": "Deep, Confident, Energetic",
        "gender": "male"
    },
    "jessica": {
        "id": "cgSgspJ2msm6clMCkdW9",
        "desc": "Playful, Bright, Warm",
        "gender": "female"
    },
    "lily": {
        "id": "pFZP5JQG7iQjIQuC4Bku",
        "desc": "Velvety Actress (British)",
        "gender": "female"
    },
    "brian": {
        "id": "nPczCjzI2devNBz1zQrb",
        "desc": "Deep, Resonant and Comforting",
        "gender": "male"
    },
    "hope": {
        "id": "tnSpp4vdxKPjI9w0GnoV",
        "desc": "Upbeat and Clear",
        "gender": "female"
    }
}

# Selected voice - Devi is recommended for Hindi/Hinglish
SELECTED_VOICE = "hope"
VOICE_ID = VOICES[SELECTED_VOICE]["id"]

# Voice settings
VOICE_SETTINGS = {
    "stability": 0.5      # 0.0-1.0: Lower = more expressive, Higher = more stable
}

# Default number of samples to generate
DEFAULT_NUM_SAMPLES = 2000


# ============================================================================
# FUNCTIONS
# ============================================================================

def generate_audio(text: str, output_path: str, voice_id: str = VOICE_ID):
    """Generate audio from text using ElevenLabs API"""
    url = f"{BASE_URL}/text-to-speech/{voice_id}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": API_KEY
    }
    
    data = {
        "text": text,
        "model_id": MODEL_ID,
        "voice_settings": VOICE_SETTINGS
    }
    
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    
    with open(output_path, "wb") as audio_file:
        audio_file.write(response.content)
    
    return output_path


def get_subscription_info():
    """Get user subscription info and remaining credits"""
    headers = {"xi-api-key": API_KEY}
    response = requests.get(f"{BASE_URL}/user/subscription", headers=headers)
    if response.status_code == 200:
        return response.json()
    return None


def read_samples(file_path: str):
    """Read text samples from file"""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]


def list_voices():
    """Print available voices"""
    print("\n=== AVAILABLE VOICES ===\n")
    for name, info in VOICES.items():
        marker = " ★ SELECTED" if name == SELECTED_VOICE else ""
        print(f"  {name:12} | {info['gender']:6} | {info['desc']}{marker}")
    print()


def main(start_idx: int = 0, end_idx: int = None):
    """Generate audio for samples from start_idx to end_idx
    
    Args:
        start_idx: Starting index (inclusive)
        end_idx: Ending index (exclusive). If None, uses start_idx as num_samples
    """
    
    # Create output directory
    output_dir = Path("audio_output")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("HINGLISH TEXT-TO-SPEECH GENERATOR")
    print("=" * 60)
    
    # Check subscription
    print("\nChecking API subscription...")
    sub_info = get_subscription_info()
    if sub_info:
        character_limit = sub_info.get("character_limit", 0)
        character_count = sub_info.get("character_count", 0)
        remaining = character_limit - character_count
        print(f"  Credits: {remaining:,} / {character_limit:,} characters")
    else:
        remaining = float('inf')
        print("  Could not check credits (proceeding anyway)")
    
    # Read samples
    all_samples = read_samples("hinglish_transliterated.txt")
    total_samples = len(all_samples)
    print(f"\nTotal samples in file: {total_samples}")
    
    # Determine range
    if end_idx is None:
        # Legacy behavior: start_idx is actually num_samples
        num_samples = start_idx if start_idx > 0 else DEFAULT_NUM_SAMPLES
        start_idx = 0
        end_idx = min(num_samples, total_samples)
    else:
        # New behavior: use start and end indices
        end_idx = min(end_idx, total_samples)
    
    # Get samples for the specified range
    samples = all_samples[start_idx:end_idx]
    print(f"Generating samples from index {start_idx} to {end_idx}")
    
    # Calculate characters needed
    total_chars = sum(len(s) for s in samples)
    print(f"Samples to generate: {len(samples)}")
    print(f"Characters needed: {total_chars:,}")
    
    if remaining != float('inf') and total_chars > remaining:
        print(f"\n⚠️  Not enough credits! Need {total_chars:,}, have {remaining:,}")
        # Recalculate max samples
        char_count = 0
        for i, sample in enumerate(samples):
            if char_count + len(sample) > remaining:
                samples = samples[:i]
                break
            char_count += len(sample)
        print(f"   Will generate: {len(samples)} samples instead")
    
    # Configuration info
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_ID}")
    print(f"  Voice: {SELECTED_VOICE} ({VOICES[SELECTED_VOICE]['desc']})")
    print(f"  Output: {output_dir}/")
    
    print("\n" + "-" * 60)
    print("Generating audio files...")
    print("-" * 60)
    
    # Generate audio for each sample
    start_time = time.time()
    for i, text in enumerate(samples):
        # Use actual index from original file in filename
        actual_idx = start_idx + i + 1
        output_file = output_dir / f"sample_{actual_idx:04d}.mp3"
        
        print(f"[{actual_idx}/{end_idx-1}] {text}")
        
        generate_audio(text, str(output_file))
        
        # Rate limiting delay
        if i < len(samples) - 1:
            time.sleep(0.3)
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print(f"COMPLETE!")
    print(f"   Generated: {len(samples)} audio files")
    print(f"   Location: {output_dir.absolute()}/")
    print(f"   Time: {elapsed:.1f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "voices":
            list_voices()
            sys.exit(0)
        elif arg == "all":
            # Generate all samples
            main(0, -1)
        else:
            start = int(sys.argv[1])
            if len(sys.argv) > 2:
                # Two arguments: start and end indices
                end = int(sys.argv[2])
                main(start, end)
            else:
                # One argument: number of samples (legacy behavior)
                main(start, None)
    else:
        # No arguments: use default
        main(0, DEFAULT_NUM_SAMPLES)