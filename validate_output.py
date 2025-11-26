import argparse
import pickle
import numpy as np
import random
from pathlib import Path
import sys

# Setup basic logging
def log(msg):
    print(f"[VALIDATOR] {msg}")

def load_data(bin_path, tok_path):
    """Loads the binary data and the tokenizer object."""
    
    # 1. Load Tokenizer
    log(f"Loading tokenizer from {tok_path}...")
    try:
        with open(tok_path, "rb") as f:
            tokenizer = pickle.load(f)
        log("✅ Tokenizer loaded successfully.")
    except Exception as e:
        log(f"❌ Failed to load tokenizer: {e}")
        sys.exit(1)

    # 2. Load Binary File (Memory Mapped)
    log(f"Loading binary data from {bin_path}...")
    try:
        if not Path(bin_path).exists():
            log(f"❌ File not found: {bin_path}")
            sys.exit(1)
            
        # IMPORTANT: Match the dtype to your writer script (uint16 or uint32)
        # We assume uint16 based on previous scripts
        data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        log(f"✅ Binary loaded. Total Tokens: {len(data):,}")
    except Exception as e:
        log(f"❌ Failed to load binary: {e}")
        sys.exit(1)

    return data, tokenizer

def validate_vocab_range(data, tokenizer):
    """Checks if any token ID in the file exceeds the tokenizer's vocabulary size."""
    
    vocab_size = None
    
    # 1. Try standard attributes
    if hasattr(tokenizer, "vocab_size"):
        vocab_size = tokenizer.vocab_size
        
    # 2. Try attributes specific to the user's Tokenizer class
    elif hasattr(tokenizer, "ttoi") and tokenizer.ttoi:
        vocab_size = len(tokenizer.ttoi)
    elif hasattr(tokenizer, "final_vocab_size"):
        vocab_size = tokenizer.final_vocab_size
        
    # 3. Fallbacks
    elif hasattr(tokenizer, "vocab"):
        vocab_size = len(tokenizer.vocab)
    elif hasattr(tokenizer, "id_to_token"):
        vocab_size = len(tokenizer.id_to_token)
    
    if vocab_size:
        max_id = np.max(data) if len(data) > 0 else 0
        log(f"🔍 Vocab Check: Max ID in file is {max_id}, Vocab Size is ~{vocab_size}")
        
        if max_id >= vocab_size:
            log(f"⚠️ WARNING: Found token ID {max_id} which is >= Vocab Size! This implies data corruption or wrong tokenizer.")
        else:
            log("✅ All tokens are within valid vocabulary range.")
    else:
        log("⚠️ Could not determine vocab size automatically. Skipping range check.")

def decode_sample(data, tokenizer, num_samples=3, sample_length=10):
    """Decodes random sequences to check for human readability."""
    
    total_len = len(data)
    if total_len == 0:
        log("File is empty.")
        return

    log("\n=== DECODING SAMPLES ===")
    
    # Helper to decode a list of IDs based on Tokenizer mode
    def decode_ids(ids):
        # Convert numpy types to native python ints for compatibility
        ids_list = ids.tolist() if hasattr(ids, "tolist") else list(ids)

        # 1. Detect Mode from User's Tokenizer Class
        code_mode = getattr(tokenizer, "code_mode", None)
        
        # BPE MODE
        if code_mode == 'bpe':
            # In BPE mode, the tokenizer has an internal text_tokenizer named 'tt'
            if hasattr(tokenizer, "tt") and hasattr(tokenizer.tt, "decode"):
                try:
                    # BPE tokenizers usually decode a sequence into a coherent string
                    return tokenizer.tt.decode(ids_list)
                except Exception as e:
                    return f"<BPE Decode Error: {e}>"
            else:
                return "<BPE mode detected but 'tt' attribute missing>"

        # CONCEPT MODE
        elif code_mode == 'concept':
            # In Concept mode, we look up IDs in the 'itot' (index-to-token) dictionary
            if hasattr(tokenizer, "itot"):
                 return [tokenizer.itot.get(i, f"<UNK:{i}>") for i in ids_list]
            else:
                return "<Concept mode detected but 'itot' attribute missing>"
        
        # FALLBACKS (For other tokenizer types or if detection fails)
        if hasattr(tokenizer, "decode"):
            return tokenizer.decode(ids_list)
        elif hasattr(tokenizer, "id_to_token"):
            return [tokenizer.id_to_token.get(i, "<UNK>") for i in ids_list]
        elif hasattr(tokenizer, "itot"):
             return [tokenizer.itot.get(i, f"<UNK:{i}>") for i in ids_list]
        
        return ["(Tokenizer has no recognizable decode method)"]

    # 1. Check Beginning
    start_ids = data[:sample_length]
    log(f"\n🔹 HEAD (First {sample_length} tokens):")
    log(f"   IDs: {start_ids}")
    log(f"   TXT: {decode_ids(start_ids)}")

    # 2. Check Random Middle Chunks
    for i in range(num_samples):
        # Pick a random spot
        idx = random.randint(sample_length, total_len - sample_length)
        chunk_ids = data[idx : idx + sample_length]
        
        log(f"\n🔹 RANDOM SAMPLE #{i+1} (Offset {idx:,}):")
        log(f"   IDs: {chunk_ids}")
        log(f"   TXT: {decode_ids(chunk_ids)}")

    # 3. Check End
    end_ids = data[-sample_length:]
    log(f"\n🔹 TAIL (Last {sample_length} tokens):")
    log(f"   IDs: {end_ids}")
    log(f"   TXT: {decode_ids(end_ids)}")
    log("\n==========================")

def main():
    parser = argparse.ArgumentParser(description="Validate Tokenized Binary File")
    parser.add_argument("--bin", required=True, help="Path to the .bin file (e.g., train_codes.bin)")
    parser.add_argument("--tok", required=True, help="Path to the tok.pkl file")
    
    args = parser.parse_args()
    
    data, tokenizer = load_data(args.bin, args.tok)
    validate_vocab_range(data, tokenizer)
    decode_sample(data, tokenizer)

if __name__ == "__main__":
    main()