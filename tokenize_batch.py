import polars as pl
import logging
from pathlib import Path
from modules.tokenizer_v2 import Tokenizer
import pickle
from datetime import datetime
import json
import yaml
from typing import Any
import argparse



##############
####config####
### MEDS format INPUT data###
def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg["paths"]["data_root"])

    cfg_out = {
        "TRAIN_FOLDER": data_root / cfg["paths"]["train_folder"],
        "VAL_FOLDER":   data_root / cfg["paths"]["val_folder"],
        "TEST_FOLDER":  data_root / cfg["paths"]["test_folder"],
        "RUN_PATH":     data_root / cfg["paths"]["run_path"],
        "N_TRAIN_FILES": cfg["train"]["n_train_files"],
        "DEFAULT_COLS":  cfg["train"]["default_cols"],
        "CONFIGS":       cfg["tokenizer_configs"],
    }
    return cfg_out

####END config ####
########################
#
#
######################
####Logger Setup######
def setup_logger(log_dir="logs", name="tokenizer", level=logging.INFO):
    Path(log_dir).mkdir(exist_ok=True)
    log_file = Path(log_dir) / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicates if setup_logger is called again
    if logger.handlers:
        # Optional: if you want to completely reset, do logger.handlers.clear()
        return logger

    # Do not bubble to root (avoids duplicate prints if root has handlers)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

logger = logging.getLogger("tokenizer")

if logger.handlers:
    print("Logger already loaded")
else:
    logger = setup_logger(name="tokenizer")
    print("Logger loaded succesffully")


####End Logger Setup###
#######################
#
#
#######################
####helper functions####
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bulk tokenization runner with YAML configs"
    )
    parser.add_argument(
        "-c", "--config",
        default="configs/config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    return parser.parse_args()

def load_as_single_df(path: str | Path, num_files: int, cols: list) -> pl.DataFrame:
    """
    Returns a single Polars dataframe with N of the parquet files at Path. Includes only cols specified in Cols.
    This is generally used to load the training data as a single dataframe for training the tokenizers.
    """
    path = Path(path)
    files = [f"{path}/{i}.parquet" for i in range(num_files)]  # 0..291
    lf = pl.scan_parquet(files).select(cols)
    df = lf.collect(engine='streaming')
    return df

def make_config_tag(config: dict) -> str:
    base = f"{config['code_mode']}_{config['num_type']}_{config['num_seq']}"
    if 'final_vocab_size' in config and config['final_vocab_size'] is not None:
        return f"{base}_{config['final_vocab_size']}"
    return base

def fix_nomenclature(df: pl.DataFrame) -> pl.DataFrame: ### NOT SURE IF WE STILL NEED THIS?? ###
    df_out = df.with_columns(
    pl.when(
        pl.col("numeric_value").is_not_null()
        & (
            pl.col("code")
            .cast(pl.Utf8)
            .str.count_matches("//")
            .fill_null(0)
            < 1
        )
    )
    .then(pl.lit("LAB//") + pl.col("code").cast(pl.Utf8))
    .otherwise(pl.col("code").cast(pl.Utf8))
    .alias("code")
    )
    return df_out

def load_and_train_tok(config:dict, train_df: pl.DataFrame, run_folder:str) -> Tokenizer:
    """
    Takes the training dataframe and a single config dictionary and saves and outputs the resulting tokenizer.
    output_path is the base folder for the tokenization run. The sub-folders for each config setting will be created and populated based on the config.
    """    
    ## Fixing issue with their nomenclature MAY BE REMOVABLE IN THE FUTURE????? ###
    tag = Path(make_config_tag(config))
    run_folder = Path(run_folder)
    save_folder = run_folder / tag
    save_folder.mkdir(parents=True, exist_ok=True)
    print(f"Training tokenizer: {tag}")
    tok = Tokenizer(**config)
    try:
        logger.info(f" ==== NEW CONFIG: Training tokenizer for {tag} ==== ")
        tok.train(train_df)
        logger.info(f" ✅ Trained tokenizer successfully for {tag} ")
        pickle_file = save_folder / f"{tag}.pkl"
        with open(pickle_file, 'wb') as outpt:
            pickle.dump(tok, outpt, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        logger.exception(f" ❌ Failed to train tokenizer {tag}: {e}")
    return tok, tag

def get_output_paths(
    save_folder: Path, input_file: Path
) -> tuple[Path, Path]:
    parent, filename = input_file.parts[-2:]
    base_dir = save_folder / parent
    event_path = base_dir / filename
    num_path = base_dir / f"num_{filename}"
    # Ensure the base dir exists (covers both)
    base_dir.mkdir(parents=True, exist_ok=True)
    return event_path, num_path

def get_files_to_tokenize(folders: list[Path]) -> list[Path]:
    files: list[Path] = []
    for folder in folders:
        files.extend(sorted(p for p in folder.iterdir() if p.suffix == ".parquet"))
    return files

def tokenize_multiple_files(files: list, tok: Tokenizer, tag: Path, run_folder: str, cols: list) -> None:
    run_folder = Path(run_folder)
    save_folder = run_folder / tag
    for f in files:
        fpath = Path(f)
        logger.info(f"Tokenizing {fpath}")
        try:
            print(f"  Tokenizing: {fpath}")
            lf = pl.scan_parquet(str(fpath)).select(cols)
            df = lf.collect(engine='streaming')
            df = fix_nomenclature(df)
            encdf = tok.encode(df)
            encdf1 = pl.DataFrame(encdf[0])
            encdf2 = pl.DataFrame(encdf[1])
            event_path, num_path = get_output_paths(save_folder, fpath)
            encdf1.write_parquet(event_path)
            logger.info(f"✅ wrote {event_path}")
            if not encdf2.is_empty():
                encdf2.write_parquet(num_path)
                logger.info(f"✅ wrote {num_path}")   

        except Exception as e:
            logger.exception(f"  failed on {fpath}: {e}")

def train_and_tokenize(config: dict, train_df: pl.DataFrame, run_folder: str | Path, files: list, cols: list) -> None:
    run_folder = Path(run_folder)
    tok, tag = load_and_train_tok(config=config, train_df=train_df, run_folder=run_folder)
    tokenize_multiple_files(files = files, tok = tok, tag=tag, run_folder=run_folder, cols=cols)

def write_run_metadata(run_folder: str | Path, cfg: dict[str, Any]) -> None:
    """
    Write a README.md and run_metadata.json into run_folder
    describing this tokenization run.
    """
    run_folder = Path(run_folder)
    run_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    meta = {
        "timestamp": timestamp,
        "train_folder": str(cfg["TRAIN_FOLDER"]),
        "val_folder": str(cfg["VAL_FOLDER"]),
        "test_folder": str(cfg["TEST_FOLDER"]),
        "run_path": str(cfg["RUN_PATH"]),
        "n_train_files": cfg["N_TRAIN_FILES"],
        "default_cols": cfg["DEFAULT_COLS"],
        "configs": cfg["CONFIGS"],
    }

    # JSON metadata
    (run_folder / "run_metadata.json").write_text(
        json.dumps(meta, indent=2)
    )

    # Human-readable README
    readme_lines = [
        "# Tokenization run",
        "",
        f"- **Timestamp**: {timestamp}",
        f"- **Train folder**: `{meta['train_folder']}`",
        f"- **Val folder**: `{meta['val_folder']}`",
        f"- **Test folder**: `{meta['test_folder']}`",
        f"- **RUN_PATH**: `{meta['run_path']}`",
        f"- **N_TRAIN_FILES**: {meta['n_train_files']}",
        f"- **DEFAULT_COLS**: `{meta['default_cols']}`",
        "",
        "## Configs",
        "",
        "```json",
        json.dumps(meta["configs"], indent=2),
        "```",
        "",
    ]
    (run_folder / "README.md").write_text("\n".join(readme_lines))

    
####END helper functions#####
############################
#
#
############
####Main####
def main():
    args = parse_args()
    cfg = load_config(args.config)

    train_path = Path(cfg["TRAIN_FOLDER"])
    val_path = Path(cfg["VAL_FOLDER"])
    test_path = Path(cfg["TEST_FOLDER"])
    run_path = Path(cfg["RUN_PATH"])
    n_train_files = cfg["N_TRAIN_FILES"]
    configs = cfg["CONFIGS"]
    cols = cfg["DEFAULT_COLS"]


    logger.info(
        "Loaded the following configs:\n%s",
        "\n".join(str(config) for config in configs)
    )

    train_df = load_as_single_df(train_path, num_files=n_train_files, cols=cols)
    train_df = fix_nomenclature(train_df)

    folders_to_tokenize = [Path(train_path), Path(val_path), Path(test_path)]
    logger.info(
        "Will tokenize all files in the following folders:\n%s",
        "\n".join(str(folder) for folder in folders_to_tokenize)
    )
    files = get_files_to_tokenize(folders_to_tokenize)
    write_run_metadata(run_path, cfg)
    for config in configs:
        train_and_tokenize(config, train_df, run_path, files, cols=cols)

if __name__ == "__main__":
    main()