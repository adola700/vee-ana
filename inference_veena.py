"""
Inference script for Veena TTS model (original or merged).

Usage:
    python inference_veena.py --model ./veena_hinglish_merged --input eval_data_25.csv --output eval_audio_merged
    python inference_veena.py --model maya-research/veena-tts --input eval_data_25.csv --output eval_audio_base
"""

import argparse
import os
import csv
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from snac import SNAC
import soundfile as sf

# Control token IDs (fixed for Veena)
START_OF_SPEECH_TOKEN = 128257
END_OF_SPEECH_TOKEN = 128258
START_OF_HUMAN_TOKEN = 128259
END_OF_HUMAN_TOKEN = 128260
START_OF_AI_TOKEN = 128261
END_OF_AI_TOKEN = 128262
AUDIO_CODE_BASE_OFFSET = 128266


def decode_snac_tokens(snac_tokens, snac_model):
    """De-interleave and decode SNAC tokens to audio."""
    if not snac_tokens or len(snac_tokens) % 7 != 0:
        return None

    snac_device = next(snac_model.parameters()).device

    # De-interleave tokens into 3 hierarchical levels
    codes_lvl = [[] for _ in range(3)]
    llm_codebook_offsets = [AUDIO_CODE_BASE_OFFSET + i * 4096 for i in range(7)]

    for i in range(0, len(snac_tokens), 7):
        # Level 0: Coarse (1 token)
        codes_lvl[0].append(snac_tokens[i] - llm_codebook_offsets[0])
        # Level 1: Medium (2 tokens)
        codes_lvl[1].append(snac_tokens[i+1] - llm_codebook_offsets[1])
        codes_lvl[1].append(snac_tokens[i+4] - llm_codebook_offsets[4])
        # Level 2: Fine (4 tokens)
        codes_lvl[2].append(snac_tokens[i+2] - llm_codebook_offsets[2])
        codes_lvl[2].append(snac_tokens[i+3] - llm_codebook_offsets[3])
        codes_lvl[2].append(snac_tokens[i+5] - llm_codebook_offsets[5])
        codes_lvl[2].append(snac_tokens[i+6] - llm_codebook_offsets[6])

    # Convert to tensors for SNAC decoder
    hierarchical_codes = []
    for lvl_codes in codes_lvl:
        tensor = torch.tensor(lvl_codes, dtype=torch.int32, device=snac_device).unsqueeze(0)
        if torch.any((tensor < 0) | (tensor > 4095)):
            raise ValueError("Invalid SNAC token values")
        hierarchical_codes.append(tensor)

    # Decode with SNAC
    with torch.no_grad():
        audio_hat = snac_model.decode(hierarchical_codes)

    return audio_hat.squeeze().clamp(-1, 1).cpu().numpy()


def generate_speech(text, model, tokenizer, snac_model, speaker="mixed_hinglish_Speaker", 
                    temperature=0.4, top_p=0.9, max_audio_seconds=30):
    """Generate speech from text using specified speaker voice."""

    # Prepare input with speaker token
    prompt = f"<spk_{speaker}> {text}"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

    # Construct full sequence: [HUMAN] <spk_speaker> text [/HUMAN] [AI] [SPEECH]
    input_tokens = [
        START_OF_HUMAN_TOKEN,
        *prompt_tokens,
        END_OF_HUMAN_TOKEN,
        START_OF_AI_TOKEN,
        START_OF_SPEECH_TOKEN
    ]

    input_ids = torch.tensor([input_tokens], device=model.device)

    # Calculate max tokens: ~86 tokens per second of audio (7 tokens * ~12 frames/sec)
    # Use text length as heuristic, but cap at max_audio_seconds
    estimated_seconds = len(text) / 15  # ~15 chars per second of speech
    max_seconds = min(estimated_seconds * 1.5, max_audio_seconds)
    max_tokens = int(max_seconds * 86) + 50  # Add buffer for control tokens

    # Generate audio tokens
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN]
        )

    # Extract SNAC tokens
    generated_ids = output[0][len(input_tokens):].tolist()
    snac_tokens = [
        token_id for token_id in generated_ids
        if AUDIO_CODE_BASE_OFFSET <= token_id < (AUDIO_CODE_BASE_OFFSET + 7 * 4096)
    ]

    if not snac_tokens:
        return None

    # Trim to complete frames (multiple of 7)
    snac_tokens = snac_tokens[:len(snac_tokens) // 7 * 7]

    # Decode audio
    audio = decode_snac_tokens(snac_tokens, snac_model)
    return audio


def load_model(model_path, use_4bit=False):
    """Load model from path (local or HuggingFace)."""
    print(f"Loading model from: {model_path}")
    
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def load_csv(csv_path):
    """Load evaluation data from CSV."""
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle different column naming conventions
            item_id = row.get('ID', row.get('id', ''))
            text = row.get('Sentence', row.get('sentence', row.get('text', row.get('hinglish', ''))))
            category = row.get('Category', row.get('category', 'unknown'))
            if item_id and text:
                rows.append({
                    'id': item_id,
                    'text': text,
                    'category': category
                })
    return rows


def main():
    parser = argparse.ArgumentParser(description='Run Veena TTS inference on evaluation data')
    parser.add_argument('--model', type=str, default='./veena_hinglish_merged',
                        help='Path to model (local path or HuggingFace model ID)')
    parser.add_argument('--input', type=str, default='eval_data_25.csv',
                        help='Input CSV file with evaluation sentences')
    parser.add_argument('--output', type=str, default='eval_audio_output',
                        help='Output directory for generated audio files')
    parser.add_argument('--speaker', type=str, default='mixed_hinglish_Speaker',
                        help='Speaker name to use for generation')
    parser.add_argument('--temperature', type=float, default=0.4,
                        help='Sampling temperature (default: 0.4)')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='Top-p sampling (default: 0.9)')
    parser.add_argument('--use-4bit', action='store_true',
                        help='Use 4-bit quantization for inference')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip files that already exist')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    print(f"Output directory: {args.output}")
    
    # Load SNAC model
    print("Loading SNAC model...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
    
    # Load Veena model
    model, tokenizer = load_model(args.model, use_4bit=args.use_4bit)
    
    # Load evaluation data
    print(f"Loading data from: {args.input}")
    rows = load_csv(args.input)
    print(f"Found {len(rows)} sentences to process")
    
    # Process each sentence
    successful = 0
    failed = 0
    
    for row in tqdm(rows, desc="Generating audio"):
        item_id = row['id']
        text = row['text']
        category = row['category'].replace(' ', '_').replace('/', '_')
        
        # Generate filename
        filename = f"{item_id}_{category}.wav"
        output_path = os.path.join(args.output, filename)
        
        # Skip if exists and flag is set
        if args.skip_existing and os.path.exists(output_path):
            print(f"Skipping {filename} (already exists)")
            successful += 1
            continue
        
        try:
            audio = generate_speech(
                text, 
                model, 
                tokenizer, 
                snac_model, 
                speaker=args.speaker,
                temperature=args.temperature,
                top_p=args.top_p
            )
            
            if audio is not None:
                sf.write(output_path, audio, 24000)
                successful += 1
            else:
                print(f"Warning: No audio generated for ID {item_id}")
                failed += 1
                
        except Exception as e:
            print(f"Error processing ID {item_id}: {e}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("Inference Complete!")
    print("=" * 50)
    print(f"Total processed: {len(rows)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {args.output}")


if __name__ == "__main__":
    main()
