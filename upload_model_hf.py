"""
Upload merged Veena model to Hugging Face Hub.

Usage:
    python upload_model_hf.py --repo-id YOUR_USERNAME/veena-hinglish --model-path ./veena_hinglish_merged
"""

import argparse
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, login, create_repo

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description='Upload model to Hugging Face Hub')
    parser.add_argument('--repo-id', type=str, required=True,
                        help='Repository ID on HuggingFace (e.g., username/model-name)')
    parser.add_argument('--model-path', type=str, default='./veena_hinglish_merged',
                        help='Path to the model directory to upload')
    parser.add_argument('--private', action='store_true',
                        help='Make the repository private')
    parser.add_argument('--commit-message', type=str, default='Upload Veena Hinglish merged model',
                        help='Commit message for the upload')
    
    args = parser.parse_args()
    
    # Verify model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist")
        return
    
    print("=" * 60)
    print("Uploading model to Hugging Face Hub")
    print("=" * 60)
    print(f"Repository: {args.repo_id}")
    print(f"Model path: {args.model_path}")
    print(f"Private: {args.private}")
    print()
    
    # Initialize API with token from environment
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN not found in environment variables")
        print("Please set HF_TOKEN in your .env file or environment")
        return
    
    # Login with token
    login(token=hf_token)
    print("Authenticated with HuggingFace")
    
    api = HfApi(token=hf_token)
    
    # Create repository if it doesn't exist
    print("[1/2] Creating/verifying repository...")
    try:
        create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True
        )
        print(f"Repository '{args.repo_id}' is ready")
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload the model folder
    print("\n[2/2] Uploading model files...")
    print("This may take a while for large models...")
    
    api.upload_folder(
        folder_path=args.model_path,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
    )
    
    print("\n" + "=" * 60)
    print("Upload Complete!")
    print("=" * 60)
    print(f"Model URL: https://huggingface.co/{args.repo_id}")
    print("\nYou can now use the model with:")
    print(f'  model = AutoModelForCausalLM.from_pretrained("{args.repo_id}")')

if __name__ == "__main__":
    main()
