import os
import csv
import json
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Configuration
# -----------------------------
BATCH_SIZE = 10    # Number of items per batch
MAX_WORKERS = 5    # Concurrent requests (keep low to avoid rate limits)
MAX_SAMPLES = float("inf")  # Total samples to process (for testing)
INPUT_CSV = "eval_data_25.csv"  # Input CSV file
OUTPUT_DIR = "generated_audio"  # Output directory for audio files
TEMP_DIR = "temp_audio_outputs"  # Temporary directory for batch tracking
MODEL = "gpt-4o-mini-tts"  # OpenAI TTS model
VOICE = "alloy"  # Voice to use

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# Helper Functions
# -----------------------------
def generate_speech(text, output_file):
    """
    Generates speech using OpenAI API and saves to file.
    Returns the output file path on success, None on failure.
    """
    try:
        output_path = Path(output_file)
        
        with client.audio.speech.with_streaming_response.create(
            model=MODEL,
            voice=VOICE,
            input=text
        ) as response:
            response.stream_to_file(output_path)
            
        return str(output_path)
    except Exception as e:
        print(f"Error generating speech for '{text[:30]}...': {e}")
        return None

def process_single_item(item):
    """Process a single item (id, text) and save audio."""
    item_id, text = item
    output_file = os.path.join(OUTPUT_DIR, f"audio_{item_id}.mp3")
    
    result = generate_speech(text, output_file)
    
    return {
        "id": item_id,
        "text": text,
        "audio_file": result,
        "success": result is not None
    }

def process_batch(batch_items, batch_id):
    """
    Process a list of items using ThreadPoolExecutor for concurrency.
    Saves batch results to a JSON file for tracking.
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {executor.submit(process_single_item, item): item for item in batch_items}
        
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"Error processing item {item[0]}: {e}")
                results.append({
                    "id": item[0],
                    "text": item[1],
                    "audio_file": None,
                    "success": False
                })
            
    output_file = os.path.join(TEMP_DIR, f"batch_{batch_id}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return len(results)

def load_csv_data(csv_file):
    """Load text data from CSV file."""
    data = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = row.get("ID", row.get("id", ""))
            text = row.get("Sentence", row.get("sentence", row.get("text", "")))
            if item_id and text:
                data.append((item_id, text))
    return data

# -----------------------------
# Main Execution
# -----------------------------
def main():
    print("Loading data from CSV...")
    
    try:
        data = load_csv_data(INPUT_CSV)
        print(f"Loaded {len(data)} items from {INPUT_CSV}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    if not data:
        print("No data found in CSV file.")
        return

    print(f"Starting audio generation. Max samples: {MAX_SAMPLES}, Batch size: {BATCH_SIZE}")

    current_batch = []
    batch_count = 0
    total_processed = 0
    
    for i, item in tqdm(enumerate(data), desc="Processing", total=min(len(data), int(MAX_SAMPLES) if MAX_SAMPLES != float("inf") else len(data))):
        if i >= MAX_SAMPLES:
            break
            
        current_batch.append(item)
        
        if len(current_batch) >= BATCH_SIZE:
            process_batch(current_batch, batch_count)
            batch_count += 1
            total_processed += len(current_batch)
            current_batch = []
            print(f"Completed batch {batch_count-1}")

    # Process remaining items
    if current_batch:
        process_batch(current_batch, batch_count)
        batch_count += 1
        total_processed += len(current_batch)
        print(f"Completed batch {batch_count-1}")

    # -----------------------------
    # Merge Outputs & Generate Report
    # -----------------------------
    print("Merging batch results...")
    all_results = []
    
    batch_files = sorted(
        glob.glob(os.path.join(TEMP_DIR, "batch_*.json")), 
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )
                     
    for bf in batch_files:
        with open(bf, "r", encoding="utf-8") as f:
            batch_data = json.load(f)
            all_results.extend(batch_data)
            
    # Save final results as JSON
    with open("audio_generation_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Count successes and failures
    successful = sum(1 for r in all_results if r["success"])
    failed = len(all_results) - successful
    
    print(f"\nAudio generation completed!")
    print(f"   Total processed: {len(all_results)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Audio files saved to: {OUTPUT_DIR}/")
    print(f"   Results saved to: audio_generation_results.json")

if __name__ == "__main__":
    main()
