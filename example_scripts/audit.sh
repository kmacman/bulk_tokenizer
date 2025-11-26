#!/bin/bash

# ===========================
# CONFIGURATION
# ===========================

# Path to the python script
SCRIPT_PATH="/nfs/roberts/project/pi_ajl89/kam385/bulk_tokenizer/audit_data.py"

# Input Raw Data Folders
DATA_ROOT="/nfs/roberts/project/pi_ajl89/is533/mimic-iv/v1/meds"
TRAIN_DIR="${DATA_ROOT}/train"
VAL_DIR="${DATA_ROOT}/val"
TEST_DIR="${DATA_ROOT}/test"

# Where to save the filtered data and the report
# This will create ./data/val_filtered and ./data/test_filtered
OUTPUT_BASE="/nfs/roberts/project/pi_ajl89/kam385/mimic-iv/v1/filtered" 

# ===========================
# EXECUTION
# ===========================

echo "Starting Data Audit and Filtration..."

python "$SCRIPT_PATH" \
    --train_dir "$TRAIN_DIR" \
    --val_dir "$VAL_DIR" \
    --test_dir "$TEST_DIR" \
    --output_base "$OUTPUT_BASE"

echo "Done! Check $OUTPUT_BASE/audit_report.txt for details."