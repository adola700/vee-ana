import os
import json
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from tqdm import tqdm
from utils import hindi_to_hinglish

# -----------------------------
# Configuration
# -----------------------------
BATCH_SIZE = 10    # Number of items per batch file
MAX_WORKERS = 10   # Concurrent requests (low to avoid rate limits)
MAX_SAMPLES = float("inf")  # Total samples to process (for testing)
TEMP_DIR = "temp_outputs"

os.makedirs(TEMP_DIR, exist_ok=True)

# -----------------------------
# Helper Functions
# -----------------------------
def process_single_item(text):
    """Wrapper to handle exceptions for a single item."""
    try:
        return hindi_to_hinglish(text)
    except Exception as e:
        print(f"Error converting '{text[:20]}...': {e}")
        return ""

def process_batch(batch_items, batch_id):
    """
    Process a list of items using ThreadPoolExecutor for concurrency within the batch.
    Saves the result to a JSON file as a dictionary: {hindi_text: hinglish_text}
    """
    results = {}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {executor.submit(process_single_item, item): item for item in batch_items}
        
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                res = future.result()
                results[item] = res
            except Exception as e:
                print(f"Error getting result for '{item[:20]}...': {e}")
                results[item] = ""
            
    output_file = os.path.join(TEMP_DIR, f"batch_{batch_id}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return len(results)

# -----------------------------
# Main Execution
# -----------------------------
def main():
    # Load dataset
    print("Loading dataset...")
    try:
        ds = load_dataset("SPRINGLab/IndicTTS-Hindi", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Starting concurrent conversion. Max samples: {MAX_SAMPLES}, Batch size: {BATCH_SIZE}")

    current_batch = []
    batch_count = 0
    total_processed = 0
    
    for i, example in tqdm(enumerate(ds), desc="Fetching Data"):
        if i >= MAX_SAMPLES:
            break
            
        current_batch.append(example["text"])
        
        if len(current_batch) >= BATCH_SIZE:
            process_batch(current_batch, batch_count)
            batch_count += 1
            total_processed += len(current_batch)
            current_batch = []
            print(f"Saved batch {batch_count-1}")

    # Process remaining
    if current_batch:
        process_batch(current_batch, batch_count)
        batch_count += 1
        total_processed += len(current_batch)
        print(f"Saved batch {batch_count-1}")

    # -----------------------------
    # Merge Outputs
    # -----------------------------
    print("Merging batch files...")
    all_data = {}
    
    # Read files in order
    batch_files = sorted(glob.glob(os.path.join(TEMP_DIR, "batch_*.json")), 
                         key=lambda x: int(x.split('_')[-1].split('.')[0]))
                         
    for bf in batch_files:
        with open(bf, "r", encoding="utf-8") as f:
            batch_data = json.load(f)
            all_data.update(batch_data)
            
    # Save final JSON as dict
    with open("hinglish_texts.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    # Save TXT (Hinglish values only, for convenience)
    with open("hinglish_texts.txt", "w", encoding="utf-8") as f:
        for hindi in all_data:
            f.write(all_data[hindi] + "\n")
            
    print(f"✅ Conversion completed. Processed {len(all_data)} items.")
    print(f"Outputs saved to hinglish_texts.json (dict) and hinglish_texts.txt (values)")

if __name__ == "__main__":
    main()
