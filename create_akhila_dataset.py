"""
Script to create and upload a Hugging Face dataset with audio and hinglish text for Akhila speaker.
Dataset will have three columns: hinglish, audio, speaker
"""

import os
from datasets import Dataset, Audio
from huggingface_hub import login

# Try to load environment variables if dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: python-dotenv not installed, using environment variables directly")

def create_hf_dataset():
    """Create a HF dataset from audio files and text."""
    
    # Get HF token - handle spaces around = in .env
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env file")
    
    # Strip whitespace and quotes from token
    hf_token = hf_token.strip().strip('"').strip("'")
    print(f"Token starts with: {hf_token[:10]}...")
    
    login(token=hf_token)
    
    # Read the hinglish texts
    text_file = "/home/jovyan/vee-ana/mixed_code.txt"
    with open(text_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Loaded {len(texts)} texts from {text_file}")
    
    # Prepare audio file paths
    audio_dir = "/home/jovyan/vee-ana/audio_output"
    audio_files = []
    valid_texts = []
    speakers = []
    
    # Line number corresponds to audio file number (1-indexed)
    for i, text in enumerate(texts, start=1):
        audio_path = os.path.join(audio_dir, f"sample_{i:04d}.mp3")
        if os.path.exists(audio_path):
            audio_files.append(audio_path)
            valid_texts.append(text)
            speakers.append("Akhila")
        else:
            print(f"Warning: Audio file not found: {audio_path}")
    
    print(f"Found {len(audio_files)} matching audio files")
    
    # Create dataset dictionary
    data = {
        "hinglish": valid_texts,
        "audio": audio_files,
        "speaker": speakers
    }
    
    # Create HF Dataset
    dataset = Dataset.from_dict(data)
    
    # Cast audio column to Audio feature
    dataset = dataset.cast_column("audio", Audio())
    
    print(f"Dataset created with {len(dataset)} samples")
    print(f"Dataset features: {dataset.features}")
    
    return dataset


def upload_dataset(dataset, repo_name="akh99/hinglish-tts-akhila"):
    """Upload the dataset to Hugging Face Hub."""
    
    print(f"\nUploading dataset to {repo_name}...")
    
    dataset.push_to_hub(
        repo_name,
        private=False
    )
    
    print(f"Dataset successfully uploaded to https://huggingface.co/datasets/{repo_name}")


if __name__ == "__main__":
    # Create the dataset
    dataset = create_hf_dataset()
    
    # Show a few samples
    print("\nFirst 3 samples from dataset:")
    for i in range(min(3, len(dataset))):
        print(f"\nSample {i+1}:")
        print(f"  Hinglish: {dataset[i]['hinglish']}")
        print(f"  Speaker: {dataset[i]['speaker']}")
    
    # Upload to HF Hub
    upload_dataset(dataset)
