"""
Script to process SPRINGLab/IndicTTS-Hindi dataset:
1. Load the IndicTTS-Hindi dataset
2. Replace/add 'hinglish' column using hinglish_texts.json mapping
3. Remove rows that don't have matching Hindi text
4. If multiple matches found, pick a random row
5. Final columns: audio, hinglish, hindi (text)
6. Upload to HuggingFace
"""

import os
import json
import random
from dotenv import load_dotenv
from datasets import load_dataset, Dataset, Audio
from huggingface_hub import login

# Load environment variables
load_dotenv()

def load_hinglish_mapping():
    """Load the Hindi to Hinglish mapping from JSON file."""
    with open("/workspace/hinglish_texts.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)
    print(f"Loaded {len(mapping)} Hindi->Hinglish mappings")
    return mapping


def process_indictts_dataset():
    """Process IndicTTS-Hindi dataset with Hinglish mappings."""
    
    # Login to HuggingFace
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env file")
    hf_token = hf_token.strip().strip('"').strip("'")
    login(token=hf_token)
    
    # Load the hinglish mapping
    hinglish_mapping = load_hinglish_mapping()
    
    # Load the IndicTTS-Hindi dataset
    print("Loading SPRINGLab/IndicTTS-Hindi dataset...")
    ds = load_dataset("SPRINGLab/IndicTTS-Hindi", split="train")
    print(f"Loaded {len(ds)} samples from IndicTTS-Hindi")
    print(f"Original columns: {ds.column_names}")
    
    # Process and filter the dataset
    matched_data = {
        "audio": [],
        "hinglish": [],
        "hindi": [],
        "speaker": []
    }
    
    # Track matches for statistics
    matched_count = 0
    unmatched_count = 0
    
    for sample in ds:
        hindi_text = sample["text"].strip()
        
        # Check if this Hindi text has a Hinglish translation
        if hindi_text in hinglish_mapping:
            hinglish_text = hinglish_mapping[hindi_text]
            # Create speaker label: gender + _indic (e.g., "male_indic" or "female_indic")
            gender_val = sample.get("gender", 0)
            # Map gender int to string (0=male, 1=female based on typical conventions)
            if isinstance(gender_val, int):
                gender = "male" if gender_val == 0 else "female"
            else:
                gender = str(gender_val).lower()
            speaker = f"{gender}_indic"
            
            matched_data["audio"].append(sample["audio"])
            matched_data["hinglish"].append(hinglish_text)
            matched_data["hindi"].append(hindi_text)
            matched_data["speaker"].append(speaker)
            matched_count += 1
        else:
            unmatched_count += 1
    
    print(f"\nMatching Statistics:")
    print(f"  Matched: {matched_count}")
    print(f"  Unmatched: {unmatched_count}")
    print(f"  Match rate: {matched_count / (matched_count + unmatched_count) * 100:.2f}%")
    
    # Handle duplicates - if multiple audio samples match the same Hindi text, pick random
    # Group by hindi text
    hindi_to_indices = {}
    for i, hindi_text in enumerate(matched_data["hindi"]):
        if hindi_text not in hindi_to_indices:
            hindi_to_indices[hindi_text] = []
        hindi_to_indices[hindi_text].append(i)
    
    # Filter to keep only one random sample per unique Hindi text
    final_data = {
        "audio": [],
        "hinglish": [],
        "hindi": [],
        "speaker": []
    }
    
    for hindi_text, indices in hindi_to_indices.items():
        # Pick a random index if multiple matches
        selected_idx = random.choice(indices)
        final_data["audio"].append(matched_data["audio"][selected_idx])
        final_data["hinglish"].append(matched_data["hinglish"][selected_idx])
        final_data["hindi"].append(matched_data["hindi"][selected_idx])
        final_data["speaker"].append(matched_data["speaker"][selected_idx])
    
    print(f"\nAfter deduplication:")
    print(f"  Unique Hindi texts: {len(final_data['hindi'])}")
    
    # Create new dataset
    new_ds = Dataset.from_dict(final_data)
    new_ds = new_ds.cast_column("audio", Audio())
    
    print(f"\nNew dataset created with {len(new_ds)} samples")
    print(f"Columns: {new_ds.column_names}")
    
    return new_ds


def upload_dataset(dataset, repo_name="akh99/indictts-hinglish"):
    """Upload the dataset to Hugging Face Hub."""
    
    print(f"\nUploading dataset to {repo_name}...")
    
    dataset.push_to_hub(
        repo_name,
        private=False
    )
    
    print(f"Dataset successfully uploaded to https://huggingface.co/datasets/{repo_name}")


if __name__ == "__main__":
    # Process the dataset
    dataset = process_indictts_dataset()
    
    # Show samples
    print("\nSample from dataset:")
    sample = dataset[0]
    print(f"  Hindi: {sample['hindi']}")
    print(f"  Hinglish: {sample['hinglish']}")
    
    # Upload to HF Hub
    upload_dataset(dataset)
