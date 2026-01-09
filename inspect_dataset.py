from datasets import load_dataset

try:
    print("Loading dataset...")
    ds = load_dataset("SPRINGLab/IndicTTS-Hindi")
    print("\nDataset Structure:")
    print(ds)
    
    print("\nFeatures:")
    for split in ds.keys():
        print(f"Split: {split}")
        print(ds[split].features)
        print("First example:", ds[split][0])
        break # Just invoke for the first split found
except Exception as e:
    print(f"Error loading dataset: {e}")
