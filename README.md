# Token_Vocab_EventExpr – Tokenization Runner

This small app trains a set of tokenizer configurations on MEDS-formatted EHR data and then applies those tokenizers to train/validation/test splits. It is designed to support experiments comparing different tokenization strategies (concept vs BPE, discrete vs continuous numeric handling, factored vs fused sequences) for downstream medical event transformer models.

The main entrypoint is a single Python script that:

1. Loads a subset of MEDS-formatted parquet files as training data.
2. Trains multiple `Tokenizer` configurations (from `modules/tokenizer_v2.py`).
3. Applies each trained tokenizer to all train/val/test files.
4. Writes tokenized outputs and the trained tokenizer object to a run directory.
5. Writes a metadata file (`run_metadata.json`) and a human-readable `README.md` into the run directory.
6. Logs progress to both stdout and a log file.

---

## Features

- **Multiple tokenizer configs in one run**
  - `code_mode`: `concept` or `bpe`
  - `num_type`: `discrete` or `continuous`
  - `num_seq`: `factored` or `fused`
  - Optional `final_vocab_size` for BPE vocab size experiments

- **MEDS-formatted input**
  - Expects parquet files with columns:
    - `subject_id`
    - `time`
    - `code`
    - `numeric_value`
    - `text_value`

- **Automatic output structure**
  - One subfolder per config under `RUN_PATH`
  - Tokenized parquet files mirroring the original train/val/test folder structure
  - Separate numeric token outputs prefixed with `num_`

- **Reproducibility helpers**
  - `run_metadata.json` with timestamp, paths, configs, and key parameters
  - Per-run `README.md` summarizing the run
  - Logging to `logs/tokenizer.log`

---

## Requirements

- Python 3.11+ (tested with 3.13)
- [Polars](https://pola.rs/)
- Any additional dependencies required by `modules/tokenizer_v2.py`
- A MEDS-formatted dataset stored as parquet files

Install dependencies (example):

```bash
module add uv # if running on Bouchet
uv sync


