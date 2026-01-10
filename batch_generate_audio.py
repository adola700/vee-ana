import os
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Configuration
# -----------------------------
BATCH_SIZE = 10       # Number of items per batch
MAX_WORKERS = 5       # Concurrent requests (keep low to avoid rate limits)
OUTPUT_DIR = "generated_audio_2"
TEMP_DIR = "temp_audio_outputs"
MODEL = "gpt-4o-mini-tts"
VOICE = "alloy"
INSTRUCTIONS = """Speak in Indian English accent. 
Manage prosody naturally with appropriate pauses, intonation, and rhythm. 
For Hinglish text (mixed Hindi-English), pronounce Hindi words authentically while maintaining natural flow. 
Use natural speech patterns with proper emphasis and pacing."""

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# -----------------------------
# Helper Functions
# -----------------------------
def generate_single_audio(item):
    """
    Generate audio for a single item.
    item: tuple of (index, text)
    Returns: tuple of (index, text, output_path, success)
    """
    idx, text = item
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    output_path = os.path.join(OUTPUT_DIR, f"audio_{idx:04d}.mp3")
    
    try:
        with client.audio.speech.with_streaming_response.create(
            model=MODEL,
            voice=VOICE,
            input=text,
            instructions=INSTRUCTIONS
        ) as response:
            response.stream_to_file(Path(output_path))
        return (idx, text, output_path, True)
    except Exception as e:
        print(f"Error generating audio for item {idx}: {e}")
        return (idx, text, None, False)


def process_batch(batch_items, batch_id):
    """
    Process a batch of items using ThreadPoolExecutor for concurrency.
    Saves the result to a JSON file with metadata.
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {executor.submit(generate_single_audio, item): item for item in batch_items}
        
        for future in as_completed(future_to_item):
            try:
                idx, text, output_path, success = future.result()
                results.append({
                    "index": idx,
                    "text": text,
                    "audio_path": output_path,
                    "success": success
                })
            except Exception as e:
                item = future_to_item[future]
                print(f"Error getting result for item {item[0]}: {e}")
                results.append({
                    "index": item[0],
                    "text": item[1],
                    "audio_path": None,
                    "success": False
                })
    
    # Sort by index to maintain order
    results.sort(key=lambda x: x["index"])
    
    output_file = os.path.join(TEMP_DIR, f"batch_{batch_id}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return len(results)


def load_texts_from_csv(csv_path, text_column="text"):
    """
    Load texts from a CSV file.
    Returns: list of (index, text) tuples
    """
    items = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if text_column in row:
                items.append((i, row[text_column]))
            elif "sentence" in row:
                items.append((i, row["sentence"]))
            else:
                # Try first column
                first_key = list(row.keys())[0]
                items.append((i, row[first_key]))
    return items


def load_texts_from_json(json_path):
    """
    Load texts from a JSON file.
    Supports dict (key-value) or list format.
    Returns: list of (index, text) tuples
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        # Use values as texts
        return [(i, text) for i, text in enumerate(data.values())]
    elif isinstance(data, list):
        return [(i, item if isinstance(item, str) else item.get("text", str(item))) 
                for i, item in enumerate(data)]
    else:
        raise ValueError("JSON must be a dict or list")


def load_texts_from_txt(txt_path):
    """
    Load texts from a plain text file (one per line).
    Returns: list of (index, text) tuples where index is 1-based (line number)
    """
    items = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            text = line.strip()
            if text:
                items.append((i, text))
    return items


def batch_generate_audio(input_path, text_column="text", max_samples=None,
                         output_dir=None, model=None, voice=None,
                         batch_size=None, max_workers=None):
    """
    Main function to batch generate audio files from a text source.
    
    Args:
        input_path: Path to CSV, JSON, or TXT file containing texts
        text_column: Column name for CSV files (default: "text")
        max_samples: Maximum number of samples to process (None for all)
        output_dir: Output directory for audio files
        model: OpenAI TTS model name
        voice: OpenAI TTS voice name
        batch_size: Number of items per batch
        max_workers: Number of concurrent workers
    
    Returns:
        List of result dictionaries with index, text, audio_path, and success status
    """
    global OUTPUT_DIR, MODEL, VOICE, BATCH_SIZE, MAX_WORKERS
    
    if output_dir:
        OUTPUT_DIR = output_dir
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    if model:
        MODEL = model
    if voice:
        VOICE = voice
    if batch_size:
        BATCH_SIZE = batch_size
    if max_workers:
        MAX_WORKERS = max_workers
    
    # Load texts based on file type
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".csv":
        items = load_texts_from_csv(input_path, text_column)
    elif ext == ".json":
        items = load_texts_from_json(input_path)
    elif ext == ".txt":
        items = load_texts_from_txt(input_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    # Limit samples if specified
    if max_samples:
        items = items[:max_samples]
    
    print(f"Loaded {len(items)} items from {input_path}")
    print(f"Starting batch audio generation. Batch size: {BATCH_SIZE}, Workers: {MAX_WORKERS}")
    print(f"Model: {MODEL}, Voice: {VOICE}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    current_batch = []
    batch_count = 0
    total_processed = 0
    
    for item in tqdm(items, desc="Processing"):
        current_batch.append(item)
        
        if len(current_batch) >= BATCH_SIZE:
            process_batch(current_batch, batch_count)
            batch_count += 1
            total_processed += len(current_batch)
            current_batch = []
            print(f"Completed batch {batch_count - 1}")
    
    # Process remaining
    if current_batch:
        process_batch(current_batch, batch_count)
        batch_count += 1
        total_processed += len(current_batch)
        print(f"Completed batch {batch_count - 1}")
    
    # -----------------------------
    # Merge Outputs
    # -----------------------------
    print("Merging batch results...")
    all_results = []
    
    import glob
    batch_files = sorted(
        glob.glob(os.path.join(TEMP_DIR, "batch_*.json")),
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )
    
    for bf in batch_files:
        with open(bf, "r", encoding="utf-8") as f:
            batch_data = json.load(f)
            all_results.extend(batch_data)
    
    # Sort by index
    all_results.sort(key=lambda x: x["index"])
    
    # Save final manifest
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # Count successes
    successful = sum(1 for r in all_results if r["success"])
    print(f"✅ Audio generation completed!")
    print(f"   Successfully generated: {successful}/{len(all_results)} files")
    print(f"   Output directory: {OUTPUT_DIR}")
    print(f"   Manifest saved to: {manifest_path}")
    
    return all_results


# -----------------------------
# Main Execution
# -----------------------------
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch generate audio files using OpenAI TTS")
    parser.add_argument("--input", default="mixed_code.txt", help="Path to input file (CSV, JSON, or TXT)")
    parser.add_argument("--text-column", default="text", help="Column name for text in CSV (default: text)")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to process")
    parser.add_argument("--output-dir", default="generated_audio_openai", help="Output directory")
    parser.add_argument("--model", default="gpt-4o-mini-tts", help="OpenAI TTS model")
    parser.add_argument("--voice", default="alloy", help="OpenAI TTS voice")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent workers")
    
    args = parser.parse_args()
    
    batch_generate_audio(
        input_path=args.input,
        text_column=args.text_column,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        model=args.model,
        voice=args.voice,
        batch_size=args.batch_size,
        max_workers=args.workers
    )


if __name__ == "__main__":
    main()
