import polars as pl
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_reference_sets(train_path: Path):
    """
    Scans the entire Training folder to build the "Ground Truth" 
    sets of allowed Codes and Subject IDs.
    """
    logger.info(f"📚 Scanning Training Data at {train_path}...")
    
    files = sorted([str(p) for p in train_path.glob("*.parquet")])
    if not files:
        raise FileNotFoundError(f"No parquet files found in {train_path}")

    # Scan and collect only unique values needed
    lf = pl.scan_parquet(files).select(["code", "subject_id"])
    df_ref = lf.collect()
    
    unique_codes = set(df_ref["code"].unique().to_list())
    unique_subjects = set(df_ref["subject_id"].unique().to_list())
    
    logger.info(f"✅ Reference loaded: {len(unique_codes):,} unique codes, {len(unique_subjects):,} unique subjects.")
    return unique_codes, unique_subjects

def process_split(
    split_name: str, 
    input_dir: Path, 
    output_dir: Path, 
    valid_codes: set, 
    train_subjects: set
):
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(list(input_dir.glob("*.parquet")))
    
    logger.info(f"🚀 Processing {split_name} ({len(files)} files)...")
    
    report_stats = {
        "files_processed": 0,
        "leaked_subjects": set(),
        "oov_codes": set(),
        "total_rows_original": 0,
        "total_rows_filtered": 0
    }
    
    # Convert to list once to ensure stability and avoid Polars warning
    valid_codes_list = list(valid_codes)

    for i, file_path in enumerate(files):
        # 1. Load File
        df = pl.read_parquet(file_path)
        report_stats["total_rows_original"] += len(df)
        
        # 2. AUDIT: Check for Subject Leakage
        file_subjects = set(df["subject_id"].unique().to_list())
        leaks = file_subjects.intersection(train_subjects)
        if leaks:
            report_stats["leaked_subjects"].update(leaks)
            logger.warning(f"  ⚠️ LEAKAGE in {file_path.name}: {len(leaks)} subjects found in Train!")

        # 3. FILTER & AUDIT: Check for OOV Codes
        file_codes = set(df["code"].unique().to_list())
        oov = file_codes - valid_codes
        if oov:
            report_stats["oov_codes"].update(oov)
        
        # Optimization: only filter if OOV codes exist
        if not oov:
            df_filtered = df
        else:
            df_filtered = df.filter(pl.col("code").is_in(valid_codes_list))

        report_stats["total_rows_filtered"] += len(df_filtered)
        report_stats["files_processed"] += 1  # <--- FIX: Increment counter

        # 4. Save Filtered File
        save_path = output_dir / file_path.name
        df_filtered.write_parquet(save_path)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(files)}...", end="\r")

    print(f"  ✅ Finished {split_name}                 ")
    return report_stats

def write_report(stats: dict, output_base: Path, train_path: Path, val_path: Path, test_path: Path):
    """Generates a text report of the audit including source paths."""
    report_path = output_base / "audit_report.txt"
    
    with open(report_path, "w") as f:
        f.write(f"DATA AUDIT REPORT - {datetime.now()}\n")
        f.write("========================================\n\n")
        
        # FIX: Add Source Paths to Report
        f.write("SOURCE DATA PATHS:\n")
        f.write(f"  - Train (Reference): {train_path.resolve()}\n")
        f.write(f"  - Validation (Raw):  {val_path.resolve()}\n")
        f.write(f"  - Test (Raw):        {test_path.resolve()}\n")
        f.write("========================================\n\n")
        
        for split, data in stats.items():
            f.write(f"SPLIT: {split}\n")
            f.write(f"  - Files Processed: {data['files_processed']}\n")
            f.write(f"  - Original Rows:   {data['total_rows_original']:,}\n")
            f.write(f"  - Filtered Rows:   {data['total_rows_filtered']:,}\n")
            f.write(f"  - OOV Codes Found: {len(data['oov_codes'])}\n")
            f.write(f"  - Subject Leakage: {len(data['leaked_subjects'])} subjects found in Train\n")
            
            if data['leaked_subjects']:
                f.write(f"    ⚠️ LEAKED IDs (First 10): {list(data['leaked_subjects'])[:10]}\n")
            
            f.write("\n")
            
    logger.info(f"📄 Report written to {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Audit and Filter Val/Test Data")
    parser.add_argument("--train_dir", type=Path, required=True)
    parser.add_argument("--val_dir", type=Path, required=True)
    parser.add_argument("--test_dir", type=Path, required=True)
    parser.add_argument("--output_base", type=Path, required=True)
    
    args = parser.parse_args()
    
    # 1. Load Train Reference
    valid_codes, train_subjects = load_reference_sets(args.train_dir)
    
    all_stats = {}
    
    # 2. Process Validation
    val_out = args.output_base / "val_filtered"
    all_stats["Validation"] = process_split("Validation", args.val_dir, val_out, valid_codes, train_subjects)
    
    # 3. Process Test
    test_out = args.output_base / "test_filtered"
    all_stats["Test"] = process_split("Test", args.test_dir, test_out, valid_codes, train_subjects)
    
    # 4. Write Report
    write_report(
        stats=all_stats, 
        output_base=args.output_base,
        train_path=args.train_dir,
        val_path=args.val_dir,
        test_path=args.test_dir
    )

if __name__ == "__main__":
    main()