import logging
import subprocess
from pathlib import Path

import librosa
import soundfile as sf

from app.config import MAX_RECORDING_SECONDS, SAMPLE_RATE

logger = logging.getLogger(__name__)


def convert_to_wav(input_path: str, output_path: str) -> str:
    try:
        audio, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
        sf.write(output_path, audio, SAMPLE_RATE)
        return output_path
    except Exception as e:
        logger.warning("librosa/soundfile conversion failed, trying ffmpeg: %s", e)

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", input_path,
                "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav", output_path,
            ],
            capture_output=True,
            check=True,
            timeout=30,
        )
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"Audio conversion failed for {input_path}: {e}") from e


def get_audio_duration(file_path: str) -> float:
    return librosa.get_duration(path=file_path)


def validate_audio(file_path: str) -> tuple[bool, str]:
    path = Path(file_path)
    if not path.exists():
        return False, "File does not exist"
    if not path.is_file():
        return False, "Path is not a file"

    try:
        duration = get_audio_duration(file_path)
    except Exception as e:
        return False, f"Cannot read audio file: {e}"

    if duration < 0.5:
        return False, "Audio is too short (minimum 0.5 seconds)"
    if duration > MAX_RECORDING_SECONDS:
        return False, f"Audio exceeds maximum duration of {MAX_RECORDING_SECONDS} seconds"

    return True, ""
