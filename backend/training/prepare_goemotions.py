"""Prepare GoEmotions dataset: map 28 fine-grained labels to 7 broad emotion classes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import PROCESSED_DIR, EMOTION_LABELS

import argparse
import logging

import pandas as pd
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

FINE_TO_BROAD = {
    2: "anger", 3: "anger", 10: "anger",
    11: "disgust",
    14: "fear", 19: "fear",
    0: "joy", 1: "joy", 4: "joy", 5: "joy", 13: "joy",
    15: "joy", 17: "joy", 18: "joy", 20: "joy", 21: "joy", 23: "joy",
    9: "sadness", 12: "sadness", 16: "sadness", 24: "sadness", 25: "sadness",
    6: "surprise", 7: "surprise", 22: "surprise", 26: "surprise",
    27: "neutral",
}

LABEL_TO_ID = {label: idx for idx, label in enumerate(EMOTION_LABELS)}


def map_example(label_indices: list[int]) -> str | None:
    """
    Map a list of fine-grained GoEmotions label indices to a single broad class.
    Returns None if labels span multiple broad classes or are empty.
    """
    if not label_indices:
        return None

    broad_labels = set()
    for idx in label_indices:
        broad = FINE_TO_BROAD.get(idx)
        if broad is None:
            return None
        broad_labels.add(broad)

    return broad_labels.pop() if len(broad_labels) == 1 else None


def process_split(dataset_split, split_name: str) -> pd.DataFrame:
    logger.info(f"Processing {split_name} split ({len(dataset_split)} examples)...")

    records = []
    dropped = 0

    for example in dataset_split:
        broad_label = map_example(example["labels"])
        if broad_label is None:
            dropped += 1
            continue
        records.append({
            "text": example["text"],
            "label": broad_label,
            "label_id": LABEL_TO_ID[broad_label],
        })

    df = pd.DataFrame(records)
    total = len(dataset_split)
    logger.info(
        f"  {split_name}: kept {len(df)}/{total} "
        f"({len(df)/total*100:.1f}%), dropped {dropped} ({dropped/total*100:.1f}%)"
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare GoEmotions for MoodMirror")
    parser.parse_args()

    output_dir = PROCESSED_DIR / "goemotions"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading GoEmotions dataset...")
    dataset = load_dataset("go_emotions", trust_remote_code=True)

    split_map = {"train": "train", "validation": "val", "test": "test"}
    all_dfs = {}

    for hf_split, out_name in split_map.items():
        df = process_split(dataset[hf_split], hf_split)
        out_path = output_dir / f"{out_name}.csv"
        df.to_csv(out_path, index=False)
        all_dfs[out_name] = df
        logger.info(f"  Saved to {out_path}")

    print("\n" + "=" * 60)
    print("GoEmotions Preparation Summary")
    print("=" * 60)

    for split_name, df in all_dfs.items():
        print(f"\n{split_name.upper()} ({len(df)} examples):")
        dist = df["label"].value_counts().sort_index()
        for label, count in dist.items():
            pct = count / len(df) * 100
            print(f"  {label:10s}  {count:6d}  ({pct:5.1f}%)")

    total_original = sum(len(dataset[s]) for s in dataset)
    total_kept = sum(len(df) for df in all_dfs.values())
    total_dropped = total_original - total_kept
    print(f"\nOverall: kept {total_kept}/{total_original}, dropped {total_dropped} "
          f"({total_dropped/total_original*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
