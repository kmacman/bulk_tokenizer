#!/bin/bash

# ==========================================
# 1. SETUP & PATHS (The "Separation")
# ==========================================
PYTHON_EXEC="/nfs/roberts/project/pi_ajl89/kam385/bulk_tokenizer/.venv/bin/python"

# --- A. PYTHON SCRIPT LOCATION ---
# Where the actual code lives
SCRIPT_PATH="/nfs/roberts/project/pi_ajl89/kam385/bulk_tokenizer/run_tokenizer.py"

# --- B. INPUT DATA (READ ONLY) ---
# The absolute path to where the raw PARQUET files live
INPUT_ROOT="/nfs/roberts/project/pi_ajl89"
# Subfolders inside that root
TRAIN_FOLDER="is533/mimic-iv/v1/meds/train"
VAL_FOLDER="kam385/mimic-iv/v1/filtered/val_filtered"
TEST_FOLDER="kam385/mimic-iv/v1/filtered/test_filtered"

# --- C. OUTPUT DATA (WRITEABLE) ---
# The absolute path to where you want the TOKENIZED files to go
OUTPUT_ROOT="/nfs/roberts/project/pi_ajl89/kam385/mimic-iv/v1/tokenized"
# The specific method name (this creates a subfolder in OUTPUT_ROOT)
METHOD_NAME="concept_discrete_fused"

# Combine them to get the final save destination
OUTPUT_DIR="${OUTPUT_ROOT}/${METHOD_NAME}"

# ==========================================
# 2. EXPERIMENT CONFIGURATION
# ==========================================

# Data Config
N_TRAIN_FILES=292
COLS="subject_id time code numeric_value"

# Tokenizer Hyperparameters
CODE_MODE="concept"
NUM_TYPE="discrete"
NUM_SEQ="fused"
#FINAL_VOCAB_SIZE=4096
#BPE_TRAINING_SAMPLE=0.1
#BPE_TRAINING_SEED=42

# ==========================================
# 3. EXECUTION
# ==========================================

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "=========================================================="
echo "🚀 Starting Tokenization Run"
echo "   Method: $METHOD_NAME"
echo "   Reading from: $INPUT_ROOT"
echo "   Writing to:   $OUTPUT_DIR"
echo "=========================================================="

"$PYTHON_EXEC" "$SCRIPT_PATH" \
    --data_root "$INPUT_ROOT" \
    --train_folder "$TRAIN_FOLDER" \
    --val_folder "$VAL_FOLDER" \
    --test_folder "$TEST_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    --n_train_files "$N_TRAIN_FILES" \
    --default_cols $COLS \
    --code_mode "$CODE_MODE" \
    --num_type "$NUM_TYPE" \
    --num_seq "$NUM_SEQ" \
#    --final_vocab_size "$FINAL_VOCAB_SIZE" \
#    --bpe_training_sample "$BPE_TRAINING_SAMPLE" \
#    --bpe_training_seed "$BPE_TRAINING_SEED"

echo "✅ Run Complete. Files saved to $OUTPUT_DIR"