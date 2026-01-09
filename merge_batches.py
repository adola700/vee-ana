import os
import glob
import json

def merge_batches(output_dir="temp_outputs", final_json="hinglish_texts.json", final_txt="hinglish_texts.txt"):
    print(f"Merging batch files from '{output_dir}'...")
    all_data = {}
    
    # Read files in order (though order doesn't strictly matter for dict, it's nice for stability)
    batch_files = sorted(glob.glob(os.path.join(output_dir, "batch_*.json")), 
                         key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    total_files = len(batch_files)
    print(f"Found {total_files} batch files.")

    for bf in batch_files:
        try:
            with open(bf, "r", encoding="utf-8") as f:
                batch_data = json.load(f)
                if isinstance(batch_data, dict):
                    all_data.update(batch_data)
                elif isinstance(batch_data, list):
                    # Should not happen based on current logic, but just in case
                    print(f"Warning: {bf} is a list, skipping or need manual handling.")
        except Exception as e:
            print(f"Error reading {bf}: {e}")
            
    # Save final JSON as dict
    print(f"Saving {len(all_data)} items to '{final_json}'...")
    with open(final_json, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    # Save TXT (Hinglish values only, for convenience)
    print(f"Saving values to '{final_txt}'...")
    with open(final_txt, "w", encoding="utf-8") as f:
        for hindi in all_data:
            f.write(all_data[hindi] + "\n")
            
    print(f"✅ Merge completed.")

if __name__ == "__main__":
    merge_batches()
