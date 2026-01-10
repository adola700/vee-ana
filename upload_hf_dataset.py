"""
Script to create and upload a Hugging Face dataset with audio and hinglish text.
Dataset will have two columns: audio, hinglish
"""

import os
from dotenv import load_dotenv
from datasets import Dataset, Audio
from huggingface_hub import login

# Load environment variables
load_dotenv()

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
    
    # Store token for later use
    global HF_TOKEN
    HF_TOKEN = hf_token
    
    # Read the hinglish texts
    text_file = "/workspace/mixed_code.txt"
    with open(text_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Loaded {len(texts)} texts from {text_file}")
    
    # Prepare audio file paths
    audio_dir = "/workspace/generated_audio_openai"
    audio_files = []
    valid_texts = []
    
    for i, text in enumerate(texts, start=1):
        audio_path = os.path.join(audio_dir, f"audio_{i:04d}.mp3")
        if os.path.exists(audio_path):
            audio_files.append(audio_path)
            valid_texts.append(text)
        else:
            print(f"Warning: Audio file not found: {audio_path}")
    
    print(f"Found {len(audio_files)} matching audio files")
    
    # Create dataset dictionary
    data = {
        "audio": audio_files,
        "hinglish": valid_texts
    }
    
    # Create HF Dataset
    dataset = Dataset.from_dict(data)
    
    # Cast audio column to Audio feature
    dataset = dataset.cast_column("audio", Audio())
    
    print(f"Dataset created with {len(dataset)} samples")
    print(f"Dataset features: {dataset.features}")
    
    return dataset


def upload_dataset(dataset, repo_name="akh99/hinglish-tts-openai"):
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
    
    # Show a sample
    print("\nSample from dataset:")
    print(dataset[0])
    
    # Upload to HF Hub
    upload_dataset(dataset)
