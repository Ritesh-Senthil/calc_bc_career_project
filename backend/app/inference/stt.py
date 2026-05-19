import logging

from app.config import WHISPER_MODEL_SIZE, WHISPER_COMPUTE_TYPE

logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel
    _FASTER_WHISPER_AVAILABLE = True
except ImportError:
    _FASTER_WHISPER_AVAILABLE = False
    logger.warning("faster-whisper not installed — transcription will be unavailable")


class STTEngine:
    def __init__(self) -> None:
        self._model: "WhisperModel | None" = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not _FASTER_WHISPER_AVAILABLE:
            return
        logger.info("Loading Whisper model (%s)…", WHISPER_MODEL_SIZE)
        self._model = WhisperModel(
            WHISPER_MODEL_SIZE, device="cpu", compute_type=WHISPER_COMPUTE_TYPE
        )
        logger.info("Whisper model loaded")

    def transcribe(self, audio_path: str) -> str:
        if not _FASTER_WHISPER_AVAILABLE:
            return "[transcription unavailable]"

        self._ensure_loaded()

        try:
            segments, _ = self._model.transcribe(audio_path, beam_size=5)
            text = " ".join(seg.text.strip() for seg in segments).strip()
            return text if text else "[empty transcription]"
        except Exception as e:
            logger.error("Transcription failed: %s", e)
            return "[transcription unavailable]"
