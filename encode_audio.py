import torch
import numpy as np

# Control token IDs (fixed for Veena)
AUDIO_CODE_BASE_OFFSET = 128266


def encode_audio_to_snac_tokens(audio, snac_model, sample_rate=24000):
    """
    Encode audio to SNAC tokens (inverse of decode_snac_tokens).
    
    Args:
        audio: Audio waveform as numpy array or torch tensor (should be in range [-1, 1])
        snac_model: The SNAC model for encoding
        sample_rate: Sample rate of the input audio (default 24000 for SNAC)
    
    Returns:
        List of interleaved SNAC tokens with LLM codebook offsets applied
    """
    # Get the device of the SNAC model
    snac_device = next(snac_model.parameters()).device
    
    # Convert numpy array to tensor if needed
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).float()
    
    # Ensure audio is in correct shape: (batch, channels, samples)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    elif audio.dim() == 2:
        audio = audio.unsqueeze(0)  # Add batch dim
    
    # Clamp audio to valid range
    audio = audio.clamp(-1, 1)
    
    # Move to SNAC device
    audio = audio.to(snac_device)
    
    # Encode with SNAC to get hierarchical codes
    with torch.no_grad():
        hierarchical_codes = snac_model.encode(audio)
    
    # hierarchical_codes is a list of 3 tensors:
    # - Level 0 (coarse): shape (batch, n_frames)
    # - Level 1 (medium): shape (batch, n_frames * 2)
    # - Level 2 (fine): shape (batch, n_frames * 4)
    
    # Extract codes as lists (remove batch dimension)
    codes_lvl0 = hierarchical_codes[0][0].cpu().tolist()  # n_frames codes
    codes_lvl1 = hierarchical_codes[1][0].cpu().tolist()  # n_frames * 2 codes
    codes_lvl2 = hierarchical_codes[2][0].cpu().tolist()  # n_frames * 4 codes
    
    n_frames = len(codes_lvl0)
    
    # Validate lengths
    if len(codes_lvl1) != n_frames * 2:
        raise ValueError(f"Level 1 codes length mismatch: expected {n_frames * 2}, got {len(codes_lvl1)}")
    if len(codes_lvl2) != n_frames * 4:
        raise ValueError(f"Level 2 codes length mismatch: expected {n_frames * 4}, got {len(codes_lvl2)}")
    
    # LLM codebook offsets for each position in the 7-token frame
    llm_codebook_offsets = [AUDIO_CODE_BASE_OFFSET + i * 4096 for i in range(7)]
    
    # Interleave tokens (reverse of de-interleave in decode_snac_tokens)
    # The decode function does:
    #   codes_lvl[0].append(snac_tokens[i] - llm_codebook_offsets[0])      # position 0
    #   codes_lvl[1].append(snac_tokens[i+1] - llm_codebook_offsets[1])    # position 1
    #   codes_lvl[1].append(snac_tokens[i+4] - llm_codebook_offsets[4])    # position 4
    #   codes_lvl[2].append(snac_tokens[i+2] - llm_codebook_offsets[2])    # position 2
    #   codes_lvl[2].append(snac_tokens[i+3] - llm_codebook_offsets[3])    # position 3
    #   codes_lvl[2].append(snac_tokens[i+5] - llm_codebook_offsets[5])    # position 5
    #   codes_lvl[2].append(snac_tokens[i+6] - llm_codebook_offsets[6])    # position 6
    
    snac_tokens = []
    
    for frame_idx in range(n_frames):
        # Position 0: Level 0, coarse (1 token per frame)
        snac_tokens.append(codes_lvl0[frame_idx] + llm_codebook_offsets[0])
        
        # Position 1: Level 1, first token (2 tokens per frame, take even index)
        snac_tokens.append(codes_lvl1[frame_idx * 2] + llm_codebook_offsets[1])
        
        # Position 2: Level 2, first token (4 tokens per frame, take index 0)
        snac_tokens.append(codes_lvl2[frame_idx * 4] + llm_codebook_offsets[2])
        
        # Position 3: Level 2, second token (4 tokens per frame, take index 1)
        snac_tokens.append(codes_lvl2[frame_idx * 4 + 1] + llm_codebook_offsets[3])
        
        # Position 4: Level 1, second token (2 tokens per frame, take odd index)
        snac_tokens.append(codes_lvl1[frame_idx * 2 + 1] + llm_codebook_offsets[4])
        
        # Position 5: Level 2, third token (4 tokens per frame, take index 2)
        snac_tokens.append(codes_lvl2[frame_idx * 4 + 2] + llm_codebook_offsets[5])
        
        # Position 6: Level 2, fourth token (4 tokens per frame, take index 3)
        snac_tokens.append(codes_lvl2[frame_idx * 4 + 3] + llm_codebook_offsets[6])
    
    return snac_tokens


def audio_to_input_tokens(audio, snac_model, sample_rate=24000):
    """
    Convert audio to input tokens suitable for the Veena model.
    This creates the full input sequence with control tokens.
    
    Args:
        audio: Audio waveform as numpy array or torch tensor
        snac_model: The SNAC model for encoding
        sample_rate: Sample rate of the input audio
    
    Returns:
        List of tokens: [START_OF_SPEECH] + snac_tokens + [END_OF_SPEECH]
    """
    START_OF_SPEECH_TOKEN = 128257
    END_OF_SPEECH_TOKEN = 128258
    
    snac_tokens = encode_audio_to_snac_tokens(audio, snac_model, sample_rate)
    
    return [START_OF_SPEECH_TOKEN] + snac_tokens + [END_OF_SPEECH_TOKEN]


# Example usage:
"""
import torch
from snac import SNAC

# Load SNAC model
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cuda")

# Load audio (should be at 24kHz sample rate)
import torchaudio
audio, sr = torchaudio.load("input.wav")
if sr != 24000:
    resampler = torchaudio.transforms.Resample(sr, 24000)
    audio = resampler(audio)

# Encode to tokens
snac_tokens = encode_audio_to_snac_tokens(audio.squeeze().numpy(), snac_model)
print(f"Generated {len(snac_tokens)} tokens ({len(snac_tokens) // 7} frames)")

# Verify by decoding back
from your_module import decode_snac_tokens
audio_reconstructed = decode_snac_tokens(snac_tokens, snac_model)
"""
