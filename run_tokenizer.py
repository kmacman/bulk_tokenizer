import polars as pl
import logging
from pathlib import Path
from meds_pipeline.meds_pipeline.tokenizers.tokenizer import Tokenizer
import pickle
import argparse
import numpy as np
import sys
import json
from datetime import datetime
import sys

# Set up logging
logger = logging.getLogger("tokenizer")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def fix_nomenclature(df: pl.DataFrame) -> pl.DataFrame:
    """Standardizes code nomenclature."""
    return df.with_columns(
        pl.when(
            pl.col("numeric_value").is_not_null() & 
            (pl.col("code").cast(pl.Utf8).str.count_matches("//").fill_null(0) < 1)
        )
        .then(pl.lit("LAB//") + pl.col("code").cast(pl.Utf8))
        .otherwise(pl.col("code").cast(pl.Utf8))
        .alias("code")
    )

def save_run_config(args, output_dir: Path):
    """Saves all script arguments to a JSON file for reproducibility."""
    config_path = output_dir / "config.json"
    
    # Convert args namespace to a dictionary
    config_dict = vars(args).copy()
    
    # Add metadata
    config_dict["timestamp"] = datetime.now().isoformat()
    config_dict["python_version"] = sys.version
    
    # Convert Path objects to strings for JSON serialization
    for key, value in config_dict.items():
        if isinstance(value, Path):
            config_dict[key] = str(value)
            
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)
        
    logger.info(f"💾 Configuration saved to {config_path}")

def process_split(split_name: str, folder_path: Path, tokenizer: Tokenizer, output_dir: Path, cols: list):
    """
    Iterates through parquets one by one, tokenizes, and APPENDS to a single .bin file.
    Low RAM usage.
    """
    code_bin_path = output_dir / f"{split_name}.bin"
    val_bin_path  = output_dir / f"{split_name}_vals.bin"
    
    # 1. CLEANUP: Delete existing files so we don't append to last run's data
    if code_bin_path.exists():
        code_bin_path.unlink()
    if val_bin_path.exists():
        val_bin_path.unlink()
    
    if not folder_path.exists():
        logger.warning(f"Folder not found: {folder_path}. Skipping.")
        return

    files = sorted([str(p) for p in folder_path.iterdir() if p.suffix == ".parquet"])
    if not files:
        logger.warning(f"No files in {folder_path}.")
        return

    logger.info(f"Processing {split_name}: {len(files)} files in Append Mode...")

    total_codes = 0
    
    # 2. ITERATE AND APPEND
    for i, fpath in enumerate(files):
        try:
            # Load ONE file into memory
            df = pl.read_parquet(fpath, columns=cols)
            df = fix_nomenclature(df)
            
            # Tokenize this chunk
            # Note: If a patient is split across files, their history will be broken here.
            enc_tuple = tokenizer.encode(df)
            
            # --- Append Codes ---
            code_df = pl.DataFrame(enc_tuple[0])
            if not code_df.is_empty():
                token_ids = code_df.get_column(code_df.columns[0])
                arr_codes = np.array(token_ids, dtype=np.uint16)
                
                # Open in 'ab' (Append Binary) mode
                with open(code_bin_path, "ab") as f:
                    f.write(arr_codes.tobytes())
                
                total_codes += len(arr_codes)

            # --- Append Values ---
            val_df = pl.DataFrame(enc_tuple[1])
            if not val_df.is_empty():
                values = val_df.get_column(val_df.columns[0])
                arr_vals = np.array(values, dtype=np.float32)
                
                with open(val_bin_path, "ab") as f:
                    f.write(arr_vals.tobytes())

            # Progress Logging
            if (i + 1) % 10 == 0:
                sys.stdout.write(f"\r  Processed {i + 1}/{len(files)} files...")
                sys.stdout.flush()

        except Exception as e:
            logger.error(f"❌ Error on file {fpath}: {e}")
            # Optional: raise e # Uncomment to stop immediately on error

    sys.stdout.write(f"\r  ✅ Finished {split_name}. Total Codes: {total_codes:,}      \n")
    sys.stdout.flush()
    
def main():
    parser = argparse.ArgumentParser(description="Single Config Tokenizer Runner")
    
    # --- Input Data Paths (Passed from Shell Script) ---
    parser.add_argument("--data_root", type=Path, required=True, help="Base folder for data")
    parser.add_argument("--train_folder", type=str, default="train")
    parser.add_argument("--val_folder", type=str, default="val")
    parser.add_argument("--test_folder", type=str, default="test")
    parser.add_argument("--n_train_files", type=int, default=50)
    
    # Accepts a list of strings for columns (e.g. --default_cols time code numeric_value)
    parser.add_argument("--default_cols", nargs='+', default=["time", "code", "numeric_value"])

    # --- Output ---
    parser.add_argument("--output_dir", type=Path, required=True)
    
    # --- Tokenizer Configs ---
    parser.add_argument("--code_mode", type=str, default="bpe")
    parser.add_argument("--num_type", type=str, default="discrete")
    parser.add_argument("--num_seq", type=str, default="factored")
    parser.add_argument("--final_vocab_size", type=int, default=4096)
    parser.add_argument("--bpe_training_sample", type=float, default=0.1)
    parser.add_argument("--bpe_training_seed", type=int, default=42)
    parser.add_argument("--split_pattern", type=str)

    args = parser.parse_args()
    
    # Construct absolute paths
    train_path = args.data_root / args.train_folder
    val_path = args.data_root / args.val_folder
    test_path = args.data_root / args.test_folder
    
    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_run_config(args, args.output_dir)

    # 1. Load Training Data subset for Tokenizer Training
    # We still perform a subset load for *training* the tokenizer to save time
    logger.info("Loading tokenizer training data...")
    train_files_all = sorted([str(p) for p in train_path.iterdir() if p.suffix == ".parquet"])
    
    # Take only the first N files for training the vocabulary
    subset_files = train_files_all[:args.n_train_files]
    
    lf = pl.scan_parquet(subset_files).select(args.default_cols)
    train_df = lf.collect()
    train_df = fix_nomenclature(train_df)

    # 2. Train Tokenizer
    tok_config = {
        "code_mode": args.code_mode,
        "num_type": args.num_type,
        "num_seq": args.num_seq,
    }
    if args.final_vocab_size:
        tok_config["final_vocab_size"] = args.final_vocab_size
    if args.bpe_training_sample:
        tok_config["bpe_training_sample"] = args.bpe_training_sample
    if args.bpe_training_seed:
        tok_config["bpe_training_seed"] = args.bpe_training_seed
    if args.split_pattern:
        if args.split_pattern == "None":
            args.split_pattern = None
        tok_config["split_pattern"] = args.split_pattern

    logger.info(f"Training tokenizer with: {tok_config}")
    tok = Tokenizer(**tok_config)
    tok.train(train_df)
    
   # Save Tokenizer
    tok_path = args.output_dir / "tok.pkl"
    with open(tok_path, 'wb') as f:
        pickle.dump(tok, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"✅ Tokenizer saved to {tok_path}")

    # 3. Process All Splits (Bulk)
    # Note: This loads the FULL train set now, not just the subset used for vocab training
    process_split("train", train_path, tok, args.output_dir, args.default_cols)
    process_split("val",   val_path,   tok, args.output_dir, args.default_cols)
    process_split("test",  test_path,  tok, args.output_dir, args.default_cols)

if __name__ == "__main__":
    main()