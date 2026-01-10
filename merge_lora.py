"""
Merge LoRA adapter weights with base Veena model.

This script loads the base Veena TTS model and merges the trained LoRA adapter,
producing a standalone model that can be used for inference without PEFT.
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configuration
BASE_MODEL_ID = "maya-research/veena-tts"
LORA_WEIGHTS_PATH = "./veena_lora_hinglish/lora_weights"
OUTPUT_DIR = "./veena_hinglish_merged"

def main():
    print("=" * 60)
    print("Merging LoRA adapter with base Veena model")
    print("=" * 60)
    
    # Load base model
    print(f"\n[1/4] Loading base model: {BASE_MODEL_ID}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load tokenizer
    print(f"\n[2/4] Loading tokenizer from: {LORA_WEIGHTS_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(LORA_WEIGHTS_PATH, trust_remote_code=True)
    
    # Load and merge LoRA adapter
    print(f"\n[3/4] Loading and merging LoRA adapter from: {LORA_WEIGHTS_PATH}")
    model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_PATH)
    
    # Merge LoRA weights into base model
    print("Merging weights...")
    merged_model = model.merge_and_unload()
    
    # Save merged model
    print(f"\n[4/4] Saving merged model to: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    merged_model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("Merge complete!")
    print(f"Merged model saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Print model size info
    total_params = sum(p.numel() for p in merged_model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # List saved files
    print("\nSaved files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  {f}: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
