# Medical Event Foundation Model - Bulk Tokenizer

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
```

## Directory Layout

``` bash
Token_Vocab_EventExpr/
├─ data/
│  ├─ raw/
│  │  ├─ train_test/
│  │  ├─ val_test/
│  │  └─ test_test/
│  └─ runs/ or test/
├─ modules/
│  └─ tokenizer_v2.py
├─ logs/
│  └─ tokenizer.log
└─ run_tokenization.py
```

## Config Files

Config files live in /configs/

By default, config/config.yaml will be loaded.

You can load a different config file by passing the -c [path.to.config.yaml] or --config [path.to.config.yaml] when you run tokenize_batch.py

By changing or using a different config.yaml file, you can change which directory files are being read from and written to, along with the configurations for the different types of tokenizers that will be trained and utilized for any given run of the tokenize_batch.py script.


## Process

1. Load training data: Loads N_TRAIN_FILES from TRAIN_FOLDER using Polars.

2. Train tokenizers: Each config creates a Tokenizer, applies fix_nomenclature(), trains on the dataframe, and saves <tag>.pkl under <RUN_PATH>/<tag>/.

3. Tokenize all splits: Collects all parquet files from train/val/test folders, tokenizes each, and writes mirrored outputs:

  <RUN_PATH>/<tag>/<split_dir>/<filename>.parquet

  <RUN_PATH>/<tag>/<split_dir>/num_<filename>.parquet (if numeric tokens exist)

## Metadata & Logging

At the start of each run, write_run_metadata() creates:

run_metadata.json — timestamp, paths, column list, and full configs.

README.md — readable summary of the run.

## Logging:

Written to logs/tokenizer.log and stdout.

Uses logger.info() for progress and logger.exception() for errors.

## Usage
``` Bash
module add uv # if you're on Bouchet and haven't already added it
uv run tokenize_batch.py
```

After completion, inspect <RUN_PATH> for:

Subfolders per config with tokenized parquet outputs.

run_metadata.json and per-run README.md.

Logs in logs/tokenizer.log.
