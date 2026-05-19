from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def get_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = get_device()

MAX_RECORDING_SECONDS = 30
MAX_FILE_SIZE_MB = 10
SAMPLE_RATE = 16000

WHISPER_MODEL_SIZE = "small"
WHISPER_COMPUTE_TYPE = "int8"

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
NUM_LABELS = len(EMOTION_LABELS)

TEXT_MODEL_DIR = MODELS_DIR / "text_emotion"
AUDIO_MODEL_DIR = MODELS_DIR / "audio_emotion"
FUSION_MODEL_DIR = MODELS_DIR / "fusion"

FALLBACK_TEXT_MODEL = "j-hartmann/emotion-english-distilroberta-base"
FALLBACK_AUDIO_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

DEFAULT_TEXT_WEIGHT = 0.6
DEFAULT_AUDIO_WEIGHT = 0.4

TEXT_BASE_MODEL = "microsoft/deberta-v3-base"
AUDIO_BASE_MODEL = "facebook/wav2vec2-base"

TRAINING_BATCH_SIZE = 16
TRAINING_EPOCHS = 5
LEARNING_RATE = 2e-5
