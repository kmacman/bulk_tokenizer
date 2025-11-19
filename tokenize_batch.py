# - [ ] TODO: Add some sort of metadata or readme file that gets saved to the run folder when ran

import polars as pl
import logging
from pathlib import Path
from modules.tokenizer_v2 import Tokenizer
import pickle
from datetime import datetime
import json

##############
####config####
### MEDS format INPUT data###
TRAIN_FOLDER = '/nfs/roberts/project/pi_ajl89/kam385/Token_Vocab_EventExpr/data/raw/train'
VAL_FOLDER = '/nfs/roberts/project/pi_ajl89/kam385/Token_Vocab_EventExpr/data/raw/val_filtered'
TEST_FOLDER = '/nfs/roberts/project/pi_ajl89/kam385/Token_Vocab_EventExpr/data/raw/test_filtered'

N_TRAIN_FILES = 292 # number of files in the train folder 

RUN_PATH = '/nfs/roberts/project/pi_ajl89/kam385/bulk_tokenizer/data/tokenized/full_tokenization_11_19_2025' #This is the path to the folder where all of the config subfolders will be saved, with the tokenized outputs and tok.pkl files

DEFAULT_COLS = ["subject_id", "time", "code", "numeric_value", "text_value"] #This shouldn't need to be changed

CONFIGS = [
    dict(code_mode='concept', num_type='discrete', num_seq='factored'),
    dict(code_mode='concept', num_type='discrete', num_seq='fused'),
    dict(code_mode='concept', num_type='continuous', num_seq='factored'),
    dict(code_mode='concept', num_type='continuous', num_seq='fused'),
    dict(code_mode='bpe', num_type='discrete', num_seq='factored',final_vocab_size=4096),
    dict(code_mode='bpe', num_type='discrete', num_seq='factored',final_vocab_size=8192),
    dict(code_mode='bpe', num_type='discrete', num_seq='factored',final_vocab_size=16384),
    dict(code_mode='bpe', num_type='continuous', num_seq='factored',final_vocab_size=4096),
    dict(code_mode='bpe', num_type='continuous', num_seq='factored',final_vocab_size=8192),
    dict(code_mode='bpe', num_type='continuous', num_seq='factored',final_vocab_size=16384),
    dict(code_mode='bpe', num_type='discrete', num_seq='factored',final_vocab_size=32768),
    dict(code_mode='bpe', num_type='continuous', num_seq='factored',final_vocab_size=32768), 
]

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

def load_as_single_df(path: str, num_files: int, cols: list = DEFAULT_COLS) -> pl.DataFrame:
    """
    Returns a single Polars dataframe with N of the parquet files at Path. Includes only cols specified in Cols.
    This is generally used to load the training data as a single dataframe for training the tokenizers.
    """
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
    train_df = fix_nomenclature(train_df)
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


def get_files_to_tokenize(folders: list) -> list:
    files_to_tokenize = []
    for folder in folders:
        files = [file for file in Path(folder).iterdir()]
        for file in files:
            files_to_tokenize.append(file)
    return files_to_tokenize

def tokenize_multiple_files(files: list, tok: Tokenizer, tag: Path, run_folder: str) -> None:
    run_folder = Path(run_folder)
    save_folder = run_folder / tag
    for f in files:
        fpath = Path(f)
        logger.info(f"Tokenizing {fpath}")
        try:
            print(f"  Tokenizing: {fpath}")
            lf = pl.scan_parquet(str(fpath)).select(DEFAULT_COLS)
            df = lf.collect(engine='streaming')
            df = fix_nomenclature(df)

            encdf = tok.encode(df)
            encdf1 = pl.DataFrame(encdf[0])
            encdf2 = pl.DataFrame(encdf[1])

            # mirror `train/0.parquet` under save_folder
            parent, filename = fpath.parts[-2:]
            out_file = save_folder / parent / filename

            num_parent = parent
            num_absolute = f"num_{filename}"
            out_file_num = save_folder / num_parent / num_absolute

            # make sure dirs exist
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file_num.parent.mkdir(parents=True, exist_ok=True)

            encdf1.write_parquet(out_file)
            if not encdf2.is_empty():
                encdf2.write_parquet(out_file_num)

            logger.info(f"✅ wrote {out_file}")
        except Exception as e:
            logger.exception(f"  failed on {fpath}: {e}")

def train_and_tokenize(config: dict, train_df: pl.DataFrame, run_folder: str | Path, files: list) -> None:
    run_folder = Path(run_folder)
    tok, tag = load_and_train_tok(config=config, train_df=train_df, run_folder=run_folder)
    tokenize_multiple_files(files = files, tok = tok, tag=tag, run_folder=run_folder)

def write_run_metadata(run_folder: str | Path, configs: list[dict]) -> None:
    """
    Write a README.md and run_metadata.json into run_folder
    describing this tokenization run.
    """
    run_folder = Path(run_folder)
    run_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    meta = {
        "timestamp": timestamp,
        "train_folder": TRAIN_FOLDER,
        "val_folder": VAL_FOLDER,
        "test_folder": TEST_FOLDER,
        "n_train_files": N_TRAIN_FILES,
        "default_cols": DEFAULT_COLS,
        "configs": CONFIGS,
    }

    # Machine-readable JSON for scripting / reproducibility
    (run_folder / "run_metadata.json").write_text(
        json.dumps(meta, indent=2)
    )

    # Human-readable README
    readme_lines = [
        "# Tokenization run",
        "",
        f"- **Timestamp**: {timestamp}",
        f"- **Train folder**: `{TRAIN_FOLDER}`",
        f"- **Val folder**: `{VAL_FOLDER}`",
        f"- **Test folder**: `{TEST_FOLDER}`",
        f"- **N_TRAIN_FILES**: {N_TRAIN_FILES}",
        f"- **DEFAULT_COLS**: `{DEFAULT_COLS}`",
        "",
        "## Configs",
        "",
        "```json",
        json.dumps(CONFIGS, indent=2),
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
    logger.info(
        "Loaded the following configs:\n%s",
        "\n".join(str(config) for config in CONFIGS)
    )
    train_df = load_as_single_df(TRAIN_FOLDER, N_TRAIN_FILES)
    folders_to_tokenize = [Path(TRAIN_FOLDER), Path(VAL_FOLDER), Path(TEST_FOLDER)]
    logger.info(
        "Will tokenize all files in the following folders:\n%s",
        "\n".join(str(folder) for folder in folders_to_tokenize)
    )
    files = get_files_to_tokenize(folders_to_tokenize)
    write_run_metadata(RUN_PATH, CONFIGS)
    for config in CONFIGS:
        train_and_tokenize(config, train_df, RUN_PATH, files)

if __name__ == "__main__":
    main()