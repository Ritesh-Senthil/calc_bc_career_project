"""Prepare RAVDESS + CREMA-D audio datasets with actor-aware stratified splits."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import DATA_DIR, PROCESSED_DIR, EMOTION_LABELS

import argparse
import logging
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABEL_TO_ID = {label: idx for idx, label in enumerate(EMOTION_LABELS)}

RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "neutral",   # calm → neutral
    "03": "joy",       # happy
    "04": "sadness",   # sad
    "05": "anger",     # angry
    "06": "fear",      # fearful
    "07": "disgust",
    "08": "surprise",  # surprised
}

CREMA_EMOTION_MAP = {
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "joy",
    "NEU": "neutral",
    "SAD": "sadness",
}


def process_ravdess(ravdess_dir: Path) -> pd.DataFrame:
    if not ravdess_dir.exists():
        logger.warning(f"RAVDESS directory not found: {ravdess_dir}")
        return pd.DataFrame()

    wav_files = list(ravdess_dir.rglob("*.wav"))
    if not wav_files:
        logger.warning("No .wav files found in RAVDESS directory")
        return pd.DataFrame()

    logger.info(f"Processing RAVDESS: {len(wav_files)} files")

    records = []
    for wav_path in wav_files:
        parts = wav_path.stem.split("-")
        if len(parts) != 7:
            continue

        emotion_code = parts[2]
        actor_id = parts[6]
        label = RAVDESS_EMOTION_MAP.get(emotion_code)
        if label is None:
            continue

        records.append({
            "file_path": str(wav_path.resolve()),
            "label": label,
            "label_id": LABEL_TO_ID[label],
            "dataset": "ravdess",
            "actor_id": f"ravdess_{actor_id}",
        })

    df = pd.DataFrame(records)
    logger.info(f"RAVDESS: {len(df)} valid samples from {df['actor_id'].nunique()} actors")
    return df


def process_crema_d(crema_dir: Path) -> pd.DataFrame:
    if not crema_dir.exists():
        logger.warning(f"CREMA-D directory not found: {crema_dir}")
        return pd.DataFrame()

    wav_files = list(crema_dir.rglob("*.wav"))
    if not wav_files:
        logger.warning("No .wav files found in CREMA-D directory")
        return pd.DataFrame()

    logger.info(f"Processing CREMA-D: {len(wav_files)} files")

    records = []
    for wav_path in wav_files:
        parts = wav_path.stem.split("_")
        if len(parts) < 3:
            continue

        actor_id = parts[0]
        emotion_code = parts[2]
        label = CREMA_EMOTION_MAP.get(emotion_code)
        if label is None:
            continue

        records.append({
            "file_path": str(wav_path.resolve()),
            "label": label,
            "label_id": LABEL_TO_ID[label],
            "dataset": "crema_d",
            "actor_id": f"crema_{actor_id}",
        })

    df = pd.DataFrame(records)
    logger.info(f"CREMA-D: {len(df)} valid samples from {df['actor_id'].nunique()} actors")
    return df


def actor_group_split(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ensuring no actor appears in more than one split."""
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(df, df["label"], groups=df["actor_id"]))

    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    relative_val_size = val_size / (1 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state)
    train_idx, val_idx = next(
        gss_val.split(df_train_val, df_train_val["label"], groups=df_train_val["actor_id"])
    )

    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

    return df_train, df_val, df_test


def oversample_surprise(df_train: pd.DataFrame, factor: int = 3) -> pd.DataFrame:
    """
    Oversample RAVDESS 'surprise' examples in the training set to compensate
    for CREMA-D having no surprise class.
    """
    surprise_mask = (df_train["label"] == "surprise") & (df_train["dataset"] == "ravdess")
    surprise_rows = df_train[surprise_mask]

    if surprise_rows.empty:
        logger.warning("No RAVDESS surprise samples found to oversample")
        return df_train

    duplicates = pd.concat([surprise_rows] * (factor - 1), ignore_index=True)
    result = pd.concat([df_train, duplicates], ignore_index=True).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    logger.info(
        f"Oversampled surprise: {len(surprise_rows)} → "
        f"{len(surprise_rows) * factor} ({factor}x)"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Prepare audio datasets for MoodMirror")
    parser.add_argument("--oversample-factor", type=int, default=3,
                        help="Oversampling multiplier for RAVDESS surprise class (default: 3)")
    args = parser.parse_args()

    output_dir = PROCESSED_DIR / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)

    ravdess_df = process_ravdess(DATA_DIR / "ravdess")
    crema_df = process_crema_d(DATA_DIR / "crema_d")

    if ravdess_df.empty and crema_df.empty:
        logger.error("No audio data found. Run download_datasets.py first.")
        print("\nPlease download at least one audio dataset:")
        print("  python download_datasets.py --dataset ravdess")
        print("  python download_datasets.py --dataset crema_d")
        return

    combined = pd.concat([ravdess_df, crema_df], ignore_index=True)
    logger.info(f"Combined dataset: {len(combined)} samples")

    df_train, df_val, df_test = actor_group_split(combined)

    train_actors = set(df_train["actor_id"])
    val_actors = set(df_val["actor_id"])
    test_actors = set(df_test["actor_id"])
    assert train_actors.isdisjoint(val_actors), "Actor leak: train ∩ val"
    assert train_actors.isdisjoint(test_actors), "Actor leak: train ∩ test"
    assert val_actors.isdisjoint(test_actors), "Actor leak: val ∩ test"

    df_train = oversample_surprise(df_train, factor=args.oversample_factor)

    for split_name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        path = output_dir / f"{split_name}.csv"
        df.to_csv(path, index=False)
        logger.info(f"Saved {split_name}: {len(df)} samples → {path}")

    print("\n" + "=" * 60)
    print("Audio Dataset Preparation Summary")
    print("=" * 60)

    for split_name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        print(f"\n{split_name.upper()} ({len(df)} samples, "
              f"{df['actor_id'].nunique()} actors):")
        dist = df["label"].value_counts().reindex(EMOTION_LABELS, fill_value=0)
        for label, count in dist.items():
            pct = count / len(df) * 100
            src = ""
            if label == "surprise":
                src = " [RAVDESS only, oversampled in train]" if split_name == "train" else " [RAVDESS only]"
            print(f"  {label:10s}  {count:6d}  ({pct:5.1f}%){src}")

    print(f"\nDataset composition:")
    for ds_name in ["ravdess", "crema_d"]:
        count = len(combined[combined["dataset"] == ds_name])
        print(f"  {ds_name:10s}  {count:6d} samples")

    print(f"\nNote: CREMA-D has no 'surprise' class. RAVDESS surprise samples "
          f"were oversampled {args.oversample_factor}x in training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
