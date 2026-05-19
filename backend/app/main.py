import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import MAX_FILE_SIZE_MB, DEVICE, TEXT_MODEL_DIR, AUDIO_MODEL_DIR, FUSION_MODEL_DIR
from app.schemas import AnalysisResponse, HealthResponse, ErrorResponse
from app.utils.audio import convert_to_wav, validate_audio
from app.inference.stt import STTEngine
from app.inference.text_emotion import TextEmotionClassifier
from app.inference.audio_emotion import AudioEmotionClassifier
from app.inference.fusion import FusionPredictor
from app.inference.response_generator import generate_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="MoodMirror API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_stt: STTEngine | None = None
_text_clf: TextEmotionClassifier | None = None
_audio_clf: AudioEmotionClassifier | None = None
_fusion: FusionPredictor | None = None


def _get_stt() -> STTEngine:
    global _stt
    if _stt is None:
        _stt = STTEngine()
    return _stt


def _get_text_clf() -> TextEmotionClassifier:
    global _text_clf
    if _text_clf is None:
        _text_clf = TextEmotionClassifier()
    return _text_clf


def _get_audio_clf() -> AudioEmotionClassifier:
    global _audio_clf
    if _audio_clf is None:
        _audio_clf = AudioEmotionClassifier()
    return _audio_clf


def _get_fusion() -> FusionPredictor:
    global _fusion
    if _fusion is None:
        _fusion = FusionPredictor()
    return _fusion


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("MoodMirror API starting — device: %s", DEVICE)
    for name, path in [
        ("Text emotion", TEXT_MODEL_DIR),
        ("Audio emotion", AUDIO_MODEL_DIR),
        ("Fusion", FUSION_MODEL_DIR),
    ]:
        if path.exists() and any(path.iterdir()):
            logger.info("%s custom model found at %s", name, path)
        else:
            logger.info("%s custom model not found — will use fallback", name)


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)) -> AnalysisResponse:
    tmp_dir = tempfile.mkdtemp(prefix="moodmirror_")

    try:
        raw_path = os.path.join(tmp_dir, file.filename or "upload.bin")
        contents = await file.read()

        if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(
                    detail=f"File exceeds {MAX_FILE_SIZE_MB} MB limit"
                ).model_dump(),
            )

        with open(raw_path, "wb") as f:
            f.write(contents)

        wav_path = os.path.join(tmp_dir, "converted.wav")
        convert_to_wav(raw_path, wav_path)

        is_valid, error_msg = validate_audio(wav_path)
        if not is_valid:
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(detail=error_msg).model_dump(),
            )

        transcript = _get_stt().transcribe(wav_path)
        text_prediction = _get_text_clf().predict(transcript)
        audio_prediction = _get_audio_clf().predict(wav_path)

        text_embedding = _get_text_clf().get_embedding(transcript)
        audio_embedding = _get_audio_clf().get_embedding(wav_path)

        fusion_prediction = _get_fusion().predict(
            text_prediction, audio_prediction, transcript,
            text_embedding=text_embedding, audio_embedding=audio_embedding,
        )

        spoken_response = generate_response(
            fusion_prediction.label,
            text_prediction.label,
            audio_prediction.label,
        )

        return AnalysisResponse(
            transcript=transcript,
            text_prediction=text_prediction,
            audio_prediction=audio_prediction,
            fusion_prediction=fusion_prediction,
            spoken_response=spoken_response,
        )

    except Exception as e:
        logger.exception("Analysis failed")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(detail=str(e)).model_dump(),
        )

    finally:
        for p in Path(tmp_dir).glob("*"):
            try:
                p.unlink()
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass
