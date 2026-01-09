################ Imports ################

# pip install transformers==4.45 accelerate hf_transfer datasets pandas peft
# pip install flash-attn --no-build-isolation

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

################ Configuration ################
MODEL_ID = "maya-research/veena-tts"
DATASET_ID = "SPRINGLab/IndicTTS-Hindi"
SPEAKER = "indictts"
LR = 2e-5
OUTPUT_DIR = "./veena_lora_fft"
MAX_SAMPLES = 10000  # Set to None for full dataset

# Control token IDs (fixed for Veena)
START_OF_SPEECH_TOKEN = 128257
END_OF_SPEECH_TOKEN = 128258
START_OF_HUMAN_TOKEN = 128259
END_OF_HUMAN_TOKEN = 128260
START_OF_AI_TOKEN = 128261
END_OF_AI_TOKEN = 128262
AUDIO_CODE_BASE_OFFSET = 128266

################ Load Model and Tokenizer ################
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    trust_remote_code=True,
)

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    print("Fixing pad token...")
    tokenizer.pad_token = tokenizer.eos_token

################ LoRA Configuration ################
print("Configuring LoRA...")

# Prepare model for LoRA training
model = prepare_model_for_kbit_training(model)

# Separate ranks for attention and FFN modules
LORA_RANK_ATTENTION = 192
LORA_RANK_FFN = 96
LORA_ALPHA_ATTENTION = 384  # 2× rank
LORA_ALPHA_FFN = 192  # 2× rank

lora_config = LoraConfig(
    r=LORA_RANK_ATTENTION,  # LoRA rank (using attention rank as default)
    lora_alpha=LORA_ALPHA_ATTENTION,  # LoRA scaling factor
    target_modules=[
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    modules_to_save=["embed_tokens"],  # Train embedding layer fully
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    # Note: PEFT doesn't natively support per-module ranks.
    # For separate attention/FFN ranks, consider using rank_pattern:
    rank_pattern={
        "q_proj": LORA_RANK_ATTENTION,
        "k_proj": LORA_RANK_ATTENTION,
        "v_proj": LORA_RANK_ATTENTION,
        "o_proj": LORA_RANK_ATTENTION,
        "gate_proj": LORA_RANK_FFN,
        "up_proj": LORA_RANK_FFN,
        "down_proj": LORA_RANK_FFN,
    },
    alpha_pattern={
        "q_proj": LORA_ALPHA_ATTENTION,
        "k_proj": LORA_ALPHA_ATTENTION,
        "v_proj": LORA_ALPHA_ATTENTION,
        "o_proj": LORA_ALPHA_ATTENTION,
        "gate_proj": LORA_ALPHA_FFN,
        "up_proj": LORA_ALPHA_FFN,
        "down_proj": LORA_ALPHA_FFN,
    },
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

################ Load Streaming Dataset ################
print("Loading streaming dataset...")
ds = load_dataset(DATASET_ID, split="train", streaming=True)

# Shuffle and take samples
if MAX_SAMPLES is not None:
    ds = ds.shuffle(seed=42, buffer_size=10000).take(MAX_SAMPLES)

################ Preprocessing Function ################
def preprocess_function(example):
    """
    Preprocess a single example for Veena TTS training.
    
    For TTS, the model learns to generate audio tokens from text input.
    Input format: [HUMAN] <spk_speaker> text [/HUMAN] [AI] [SPEECH] audio_tokens [/SPEECH] [/AI]
    """
    text = example["text"]
    
    # Note: For actual TTS training, you need audio tokens from the SNAC encoder.
    # This template shows the text preprocessing part.
    # You'll need to add audio tokenization based on your audio data.
    
    # Create the prompt with speaker token
    prompt = f"<spk_{SPEAKER}> {text}"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    
    # Construct input sequence: [HUMAN] <spk_speaker> text [/HUMAN] [AI] [SPEECH]
    input_tokens = [
        START_OF_HUMAN_TOKEN,
        *prompt_tokens,
        END_OF_HUMAN_TOKEN,
        START_OF_AI_TOKEN,
        START_OF_SPEECH_TOKEN,
    ]
    
    # For demonstration: we'd typically add audio tokens here from SNAC encoding
    # audio_tokens = encode_audio_with_snac(example["audio"])  # You'd implement this
    # full_sequence = input_tokens + audio_tokens + [END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN]
    
    # For now, this creates a placeholder that you'll need to extend
    # with actual audio tokenization
    full_sequence = input_tokens + [END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN]
    
    # Labels: mask the input prompt, only train on audio tokens
    labels = [-100] * len(input_tokens) + [END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN]
    
    attention_mask = [1] * len(full_sequence)
    
    return {
        "input_ids": full_sequence,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def preprocess_with_audio(example):
    """
    Full preprocessing function that includes audio tokenization.
    Requires SNAC model to be loaded for audio encoding.
    
    This function should be used when you have audio data available.
    """
    from snac import SNAC
    import numpy as np
    
    text = example["text"]
    audio_array = example["audio"]["array"]
    sample_rate = example["audio"]["sampling_rate"]
    
    # Load SNAC model (do this once globally in practice)
    # snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
    
    # Resample if needed (SNAC uses 24kHz)
    # if sample_rate != 24000:
    #     audio_array = resample_audio(audio_array, sample_rate, 24000)
    
    # Encode audio to SNAC tokens
    # snac_tokens = encode_with_snac(snac_model, audio_array)
    
    # Create the prompt with speaker token
    prompt = f"<spk_{SPEAKER}> {text}"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    
    # Construct full sequence
    input_tokens = [
        START_OF_HUMAN_TOKEN,
        *prompt_tokens,
        END_OF_HUMAN_TOKEN,
        START_OF_AI_TOKEN,
        START_OF_SPEECH_TOKEN,
    ]
    
    # Add audio tokens (placeholder - implement SNAC encoding)
    # audio_token_ids = [AUDIO_CODE_BASE_OFFSET + t for t in snac_tokens]
    audio_token_ids = []  # Replace with actual SNAC-encoded tokens
    
    full_sequence = input_tokens + audio_token_ids + [END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN]
    
    # Labels: mask the input, train on audio + end tokens
    labels = [-100] * len(input_tokens) + audio_token_ids + [END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN]
    
    attention_mask = [1] * len(full_sequence)
    
    # Truncate if too long
    max_length = 20512
    if len(full_sequence) > max_length:
        full_sequence = full_sequence[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
    
    return {
        "input_ids": full_sequence,
        "attention_mask": attention_mask,
        "labels": labels,
    }


################ Prepare Dataset ################
print("Preparing dataset with streaming...")

# Map the preprocessing function
tokenized_dataset = ds.map(preprocess_function, remove_columns=["audio", "text"])

################ Training Arguments ################
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    report_to="wandb",
    save_strategy="steps",
    save_steps=500,
    evaluation_strategy="no",
    # Optimizer configuration
    learning_rate=1e-4,  # Peak learning rate
    optim="adamw_8bit",  # 8-bit AdamW
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-5,
    # Batch configuration
    per_device_train_batch_size=8,  # Micro batch size
    gradient_accumulation_steps=32,  # Effective batch size = 8 * 32 = 256 (single GPU)
    num_train_epochs=1,
    max_steps=1000,  # Use max_steps for streaming datasets
    logging_dir="./logs",
    logging_steps=10,
    bf16=True,
    # LR scheduler
    warmup_ratio=0.02,
    lr_scheduler_type="cosine",
    # Streaming dataset specific
    dataloader_pin_memory=False,
    remove_unused_columns=False,
)

# Determine padding behavior based on batch size
do_padding = training_args.per_device_train_batch_size > 1

################ Initialize Trainer ################
print("Initializing Trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=do_padding,
        pad_to_multiple_of=8 if do_padding else None,
    ),
)

################ Print Model Info ################
trainer.accelerator.print(f"Model: {trainer.model}")
trainer.accelerator.print(f"Trainable parameters:")
model.print_trainable_parameters()

################ Start Training ################
print("Starting training...")
trainer.train()

################ Save Model ################
print("Saving LoRA weights...")
model.save_pretrained(f"{OUTPUT_DIR}/lora_weights")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_weights")

print("Training complete!")
