"""Download and organize datasets for MoodMirror training."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import DATA_DIR

import argparse
import io
import logging
import os
import shutil
import urllib.request
import zipfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class _DownloadProgressBar:
    def __init__(self):
        self._bar = None

    def __call__(self, block_num, block_size, total_size):
        try:
            from tqdm import tqdm
        except ImportError:
            return
        if self._bar is None:
            self._bar = tqdm(total=total_size, unit="iB", unit_scale=True)
        downloaded = block_num * block_size
        self._bar.update(block_size)
        if downloaded >= total_size and self._bar:
            self._bar.close()


def download_goemotions():
    """Verify GoEmotions is loadable via HuggingFace datasets (auto-cached)."""
    logger.info("Checking GoEmotions dataset...")
    try:
        from datasets import load_dataset

        ds = load_dataset("go_emotions", split="train", trust_remote_code=True)
        logger.info(f"GoEmotions loaded successfully: {len(ds)} training examples")
        return True
    except Exception as e:
        logger.error(f"Failed to load GoEmotions: {e}")
        logger.error("Install the datasets library: pip install datasets")
        return False


def download_ravdess():
    """Download RAVDESS speech audio from Zenodo."""
    ravdess_dir = DATA_DIR / "ravdess"
    ravdess_dir.mkdir(parents=True, exist_ok=True)

    existing_wavs = list(ravdess_dir.rglob("*.wav"))
    if existing_wavs:
        logger.info(f"RAVDESS already downloaded: {len(existing_wavs)} .wav files found")
        return True

    url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
    zip_path = DATA_DIR / "ravdess_download.zip"

    logger.info(f"Downloading RAVDESS from {url}...")
    try:
        urllib.request.urlretrieve(url, str(zip_path), _DownloadProgressBar())
    except Exception as e:
        logger.error(f"Download failed: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return False

    logger.info("Extracting RAVDESS archive...")
    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(ravdess_dir))
    except zipfile.BadZipFile:
        logger.error("Downloaded file is not a valid zip archive")
        zip_path.unlink()
        return False

    zip_path.unlink()

    extracted_wavs = list(ravdess_dir.rglob("*.wav"))
    logger.info(f"RAVDESS extracted: {len(extracted_wavs)} .wav files")
    return len(extracted_wavs) > 0


def download_crema_d():
    """
    Attempt to download CREMA-D audio WAV files.

    CREMA-D is hosted on GitHub LFS which makes bulk downloads unreliable.
    Falls back to printing manual download instructions.
    """
    crema_dir = DATA_DIR / "crema_d"
    crema_dir.mkdir(parents=True, exist_ok=True)

    existing_wavs = list(crema_dir.rglob("*.wav"))
    if existing_wavs:
        logger.info(f"CREMA-D already available: {len(existing_wavs)} .wav files found")
        return True

    repo_zip_url = "https://github.com/CheyneyComputerScience/CREMA-D/archive/refs/heads/master.zip"
    zip_path = DATA_DIR / "crema_d_download.zip"

    logger.info("Attempting to download CREMA-D from GitHub...")
    try:
        urllib.request.urlretrieve(repo_zip_url, str(zip_path), _DownloadProgressBar())

        with zipfile.ZipFile(str(zip_path), "r") as zf:
            audio_members = [m for m in zf.namelist() if "AudioWAV" in m and m.endswith(".wav")]
            if audio_members:
                logger.info(f"Found {len(audio_members)} WAV files in archive, extracting...")
                for member in audio_members:
                    filename = Path(member).name
                    with zf.open(member) as src, open(crema_dir / filename, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                zip_path.unlink()
                logger.info(f"CREMA-D extracted: {len(audio_members)} .wav files")
                return True
            else:
                logger.warning("ZIP downloaded but no WAV files found (GitHub LFS placeholders)")
                zip_path.unlink()

    except Exception as e:
        logger.warning(f"Automatic download failed: {e}")
        if zip_path.exists():
            zip_path.unlink()

    print("\n" + "=" * 70)
    print("MANUAL DOWNLOAD REQUIRED: CREMA-D")
    print("=" * 70)
    print("CREMA-D audio files are hosted via GitHub LFS and may not download")
    print("automatically. Please download manually:\n")
    print("1. Go to: https://github.com/CheyneyComputerScience/CREMA-D")
    print("2. Navigate to the AudioWAV/ folder")
    print("3. Download all .wav files")
    print(f"4. Place them in: {crema_dir.resolve()}\n")
    print("Alternative: Clone the repo with Git LFS:")
    print("  git lfs install")
    print("  git clone https://github.com/CheyneyComputerScience/CREMA-D.git")
    print(f"  cp CREMA-D/AudioWAV/*.wav {crema_dir.resolve()}/")
    print("=" * 70 + "\n")
    return False


def download_meld():
    """Download MELD CSV annotations and attempt to fetch video files."""
    meld_dir = DATA_DIR / "meld"
    meld_dir.mkdir(parents=True, exist_ok=True)

    csv_urls = {
        "train_sent_emo.csv": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv",
        "dev_sent_emo.csv": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv",
        "test_sent_emo.csv": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv",
    }

    csvs_ok = True
    for filename, url in csv_urls.items():
        csv_path = meld_dir / filename
        if csv_path.exists():
            logger.info(f"MELD {filename} already exists")
            continue

        logger.info(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, str(csv_path), _DownloadProgressBar())
            logger.info(f"Saved {filename}")
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            csvs_ok = False

    for split in ("train", "dev", "test"):
        split_dir = meld_dir / split
        split_dir.mkdir(exist_ok=True)

    video_available = any(
        list((meld_dir / split).glob("*.mp4"))
        for split in ("train", "dev", "test")
    )

    if video_available:
        total_videos = sum(
            len(list((meld_dir / s).glob("*.mp4"))) for s in ("train", "dev", "test")
        )
        logger.info(f"MELD video files already present: {total_videos} total")
    else:
        _try_download_meld_videos(meld_dir)

    return csvs_ok


def _try_download_meld_videos(meld_dir: Path):
    """Attempt gdown for MELD video files, falling back to manual instructions."""
    try:
        import gdown

        gdrive_ids = {
            "train": "1P8KuGLFSAr18W1-Ysfetmg8FkZJQpS8x",
            "dev": "1ilk1YKjaUN9DPPWjfPBAN7LtAZbtsbiY",
            "test": "1Gko-GnBiu5cVMC7hWA2-Jq8cWbfNCWzh",
        }

        for split, file_id in gdrive_ids.items():
            split_dir = meld_dir / split
            tar_path = meld_dir / f"{split}.tar.gz"
            if list(split_dir.glob("*.mp4")):
                continue

            logger.info(f"Downloading MELD {split} videos via gdown...")
            try:
                gdown.download(id=file_id, output=str(tar_path), quiet=False)
                if tar_path.exists():
                    import tarfile

                    with tarfile.open(str(tar_path), "r:gz") as tf:
                        tf.extractall(str(split_dir))
                    tar_path.unlink()
                    logger.info(f"Extracted MELD {split} videos")
            except Exception as e:
                logger.warning(f"gdown failed for {split}: {e}")
                if tar_path.exists():
                    tar_path.unlink()

    except ImportError:
        pass

    total_videos = sum(
        len(list((meld_dir / s).glob("*.mp4"))) for s in ("train", "dev", "test")
    )
    if total_videos == 0:
        print("\n" + "=" * 70)
        print("MANUAL DOWNLOAD REQUIRED: MELD Video Files")
        print("=" * 70)
        print("MELD video files need to be downloaded from Google Drive.\n")
        print("Option 1: Install gdown and re-run this script:")
        print("  pip install gdown\n")
        print("Option 2: Download manually from the MELD GitHub repo:")
        print("  https://github.com/declare-lab/MELD")
        print("  Follow their download links for video .mp4 files.\n")
        print(f"Place files in:")
        print(f"  Train: {meld_dir.resolve()}/train/dia*_utt*.mp4")
        print(f"  Dev:   {meld_dir.resolve()}/dev/dia*_utt*.mp4")
        print(f"  Test:  {meld_dir.resolve()}/test/dia*_utt*.mp4")
        print("=" * 70 + "\n")


def print_summary():
    """Print a summary of what data is currently available."""
    print("\n" + "=" * 60)
    print("DATASET AVAILABILITY SUMMARY")
    print("=" * 60)

    checks = [
        ("GoEmotions", None),
        ("RAVDESS", DATA_DIR / "ravdess"),
        ("CREMA-D", DATA_DIR / "crema_d"),
        ("MELD CSVs", DATA_DIR / "meld"),
        ("MELD Videos", None),
    ]

    for name, path in checks:
        if name == "GoEmotions":
            try:
                from datasets import load_dataset

                load_dataset("go_emotions", split="train[:1]", trust_remote_code=True)
                status = "READY (cached)"
            except Exception:
                status = "NOT AVAILABLE"
        elif name == "MELD Videos":
            meld_dir = DATA_DIR / "meld"
            count = sum(
                len(list((meld_dir / s).glob("*.mp4")))
                for s in ("train", "dev", "test")
                if (meld_dir / s).exists()
            )
            status = f"READY ({count} files)" if count > 0 else "NOT AVAILABLE"
        elif path and path.exists():
            if name == "MELD CSVs":
                csvs = list(path.glob("*.csv"))
                status = f"READY ({len(csvs)} CSVs)" if csvs else "NOT AVAILABLE"
            else:
                wavs = list(path.rglob("*.wav"))
                status = f"READY ({len(wavs)} files)" if wavs else "NOT AVAILABLE"
        else:
            status = "NOT AVAILABLE"

        indicator = "✓" if "READY" in status else "✗"
        print(f"  {indicator} {name:15s} {status}")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets for MoodMirror training")
    parser.add_argument(
        "--dataset",
        choices=["all", "goemotions", "ravdess", "crema_d", "meld"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    args = parser.parse_args()

    downloaders = {
        "goemotions": download_goemotions,
        "ravdess": download_ravdess,
        "crema_d": download_crema_d,
        "meld": download_meld,
    }

    targets = downloaders if args.dataset == "all" else {args.dataset: downloaders[args.dataset]}

    results = {}
    for name, func in targets.items():
        logger.info(f"--- Processing: {name.upper()} ---")
        results[name] = func()

    print_summary()
