import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from snac import SNAC
import soundfile as sf
import csv
import os
import tqdm
import argparse

# --- Configuration ---
OUTPUT_DIR = "generated_audio_new_model"
CSV_FILE = "eval_data_4.csv"

# Control token IDs (fixed for Veena)
START_OF_SPEECH_TOKEN = 128257
END_OF_SPEECH_TOKEN = 128258
START_OF_HUMAN_TOKEN = 128259
END_OF_HUMAN_TOKEN = 128260
START_OF_AI_TOKEN = 128261
END_OF_AI_TOKEN = 128262
AUDIO_CODE_BASE_OFFSET = 128266

def decode_snac_tokens(snac_tokens, snac_model):
    """De-interleave and decode SNAC tokens to audio"""
    if not snac_tokens or len(snac_tokens) % 7 != 0:
        return None

    # Get the device of the SNAC model
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

def generate_speech(text, model, tokenizer, snac_model, speaker="kavya", temperature=0.4, top_p=0.9):
    """Generate speech from text using specified speaker voice"""

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

    # Calculate max tokens based on text length
    max_tokens = min(int(len(text) * 1.3) * 7 + 21, 700)

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
        print(f"Warning: No audio tokens generated for text: {text}")
        return None

    # Decode audio
    audio = decode_snac_tokens(snac_tokens, snac_model)
    return audio

def main(model_name="maya-research/veena-tts"):
    print(f"Loading models from: {model_name}")
    # SNAC
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()

    # Veena TTS
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Read CSV and Process
    rows = []
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    print(f"Found {len(rows)} sentences to process.")
    
    for row in tqdm.tqdm(rows):
        idx = row['ID']
        category = row.get('Category', 'Unknown')
        # sentence = """Bhai, city ki fast life mein sapne chase karna padta hai full speed se. Subah uthke coffee piyo, laptop kholo aur code likho jaise boss.Traffic jam mein bhi mind mein ideas ka traffic chal raha hota hai nonstop. Friends bolte hain "yaar, chill kar le," but hustle hi life hai ab.Weekend pe gym jaake body banao, phir Netflix binge for recharge.Paise kamane ka plan solid hai, investments smartly karo long term.Family ko proud feel karwao, success ki story ban jao legend.End mein, dil se dil tak, keep grinding bro, sky's the limit!"""
        sentence = "Doctor ne kaha hai ki junk food avoid karo for at least a week"
        # Determine unique filename
        filename = f"{idx}_{category.replace(' ', '_')}.mp3"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        if os.path.exists(output_path):
            print(f"Skipping {filename}, already exists.")
            continue
            
        print(f"Processing ID {idx}: {sentence[:30]}...")
        
        speaker = "kavya" 
        
        try:
            audio = generate_speech(sentence, model, tokenizer, snac_model, speaker=speaker)
            if audio is not None:
                sf.write(output_path, audio, 24000)
        except Exception as e:
            print(f"Error processing ID {idx}: {e}")

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TTS inference with Veena model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="maya-research/veena-tts",
        help="HuggingFace model name for the TTS model"
    )
    args = parser.parse_args()
    main(model_name=args.model_name)

