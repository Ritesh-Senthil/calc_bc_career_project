"""Prepare MELD dataset: extract audio from videos and align with text annotations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import DATA_DIR, PROCESSED_DIR, EMOTION_LABELS, SAMPLE_RATE

import argparse
import logging
import subprocess

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABEL_TO_ID = {label: idx for idx, label in enumerate(EMOTION_LABELS)}

MELD_SPLITS = {
    "train": "train_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv",
}


def check_ffmpeg() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def extract_audio(video_path: Path, audio_path: Path) -> bool:
    """Extract mono 16kHz WAV from an MP4 file using ffmpeg."""
    if audio_path.exists():
        return True

    audio_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                "ffmpeg", "-i", str(video_path),
                "-vn", "-acodec", "pcm_s16le",
                "-ar", str(SAMPLE_RATE), "-ac", "1",
                str(audio_path),
                "-y", "-loglevel", "error",
            ],
            capture_output=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"ffmpeg failed for {video_path.name}: {e.stderr.decode().strip()}")
        return False


def process_split(
    split_name: str,
    csv_filename: str,
    meld_dir: Path,
    output_dir: Path,
    has_ffmpeg: bool,
) -> pd.DataFrame | None:
    csv_path = meld_dir / csv_filename
    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        logger.error("Run download_datasets.py --dataset meld first")
        return None

    logger.info(f"Processing MELD {split_name} from {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = {"Utterance", "Emotion", "Dialogue_ID", "Utterance_ID"}
    if not required_cols.issubset(df.columns):
        logger.error(f"Missing columns in {csv_filename}. Found: {list(df.columns)}")
        return None

    video_dir = meld_dir / split_name
    audio_out_dir = output_dir / "audio" / split_name
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    missing_videos = 0
    extract_failures = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"MELD {split_name}"):
        emotion = str(row["Emotion"]).strip().lower()
        if emotion not in LABEL_TO_ID:
            continue

        dia_id = row["Dialogue_ID"]
        utt_id = row["Utterance_ID"]
        video_name = f"dia{dia_id}_utt{utt_id}.mp4"
        video_path = video_dir / video_name

        audio_name = f"dia{dia_id}_utt{utt_id}.wav"
        audio_path = audio_out_dir / audio_name

        audio_ok = False
        if video_path.exists() and has_ffmpeg:
            audio_ok = extract_audio(video_path, audio_path)
            if not audio_ok:
                extract_failures += 1
        elif not video_path.exists():
            missing_videos += 1

        utterance_text = str(row["Utterance"]).strip()
        utterance_text = utterance_text.replace("\u2019", "'").replace("\u2018", "'")

        records.append({
            "utterance_text": utterance_text,
            "audio_path": str(audio_path.resolve()) if audio_ok else "",
            "label": emotion,
            "label_id": LABEL_TO_ID[emotion],
            "dialogue_id": dia_id,
            "utterance_id": utt_id,
        })

    result_df = pd.DataFrame(records)

    if missing_videos > 0:
        logger.warning(
            f"  {split_name}: {missing_videos} video files missing "
            f"(audio extraction skipped for these)"
        )
    if extract_failures > 0:
        logger.warning(f"  {split_name}: {extract_failures} audio extraction failures")

    return result_df


def main():
    parser = argparse.ArgumentParser(description="Prepare MELD dataset for MoodMirror")
    parser.add_argument("--skip-audio", action="store_true",
                        help="Skip audio extraction, process text annotations only")
    args = parser.parse_args()

    meld_dir = DATA_DIR / "meld"
    output_dir = PROCESSED_DIR / "meld"
    output_dir.mkdir(parents=True, exist_ok=True)

    has_ffmpeg = False
    if not args.skip_audio:
        has_ffmpeg = check_ffmpeg()
        if not has_ffmpeg:
            logger.warning(
                "ffmpeg not found. Install it to extract audio from videos.\n"
                "  macOS: brew install ffmpeg\n"
                "Proceeding with text-only processing."
            )

    all_dfs = {}
    for split_name, csv_filename in MELD_SPLITS.items():
        df = process_split(split_name, csv_filename, meld_dir, output_dir, has_ffmpeg)
        if df is None:
            continue

        out_path = output_dir / f"{split_name}.csv"
        df.to_csv(out_path, index=False)
        all_dfs[split_name] = df
        logger.info(f"Saved {split_name}: {len(df)} rows → {out_path}")

    if not all_dfs:
        logger.error("No MELD splits processed. Ensure CSVs are downloaded.")
        return

    print("\n" + "=" * 60)
    print("MELD Preparation Summary")
    print("=" * 60)

    for split_name, df in all_dfs.items():
        has_audio = (df["audio_path"] != "").sum()
        print(f"\n{split_name.upper()} ({len(df)} utterances, {has_audio} with audio):")
        dist = df["label"].value_counts().reindex(EMOTION_LABELS, fill_value=0)
        for label, count in dist.items():
            pct = count / len(df) * 100
            print(f"  {label:10s}  {count:6d}  ({pct:5.1f}%)")

    total = sum(len(df) for df in all_dfs.values())
    total_audio = sum((df["audio_path"] != "").sum() for df in all_dfs.values())
    print(f"\nTotal: {total} utterances, {total_audio} with extracted audio")

    if total_audio == 0:
        print("\nNo audio was extracted. Ensure video files are downloaded and ffmpeg is installed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
