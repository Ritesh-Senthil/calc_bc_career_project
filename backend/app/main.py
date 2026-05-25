import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import MAX_FILE_SIZE_MB, DEVICE, TEXT_MODEL_DIR
from app.schemas import AnalysisResponse, HealthResponse, ErrorResponse
from app.utils.audio import convert_to_wav, validate_audio
from app.inference.stt import STTEngine
from app.inference.text_emotion import TextEmotionClassifier
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


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("MoodMirror API starting — device: %s", DEVICE)
    if TEXT_MODEL_DIR.exists() and any(TEXT_MODEL_DIR.iterdir()):
        logger.info("Custom text model found at %s", TEXT_MODEL_DIR)
    else:
        logger.info("Custom text model not found — will use fallback")


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze(
    file: UploadFile | None = File(None),
    transcript: str | None = Form(None),
) -> AnalysisResponse:
    has_audio = file is not None and file.filename
    has_text = transcript and transcript.strip()

    if not has_audio and not has_text:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(detail="Provide an audio file or text").model_dump(),
        )

    tmp_dir = tempfile.mkdtemp(prefix="moodmirror_")

    try:
        if has_text:
            final_transcript = transcript.strip()
            logger.info("Using provided transcript (%d chars)", len(final_transcript))
        elif has_audio:
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

            final_transcript = _get_stt().transcribe(wav_path)
            logger.info("Used Whisper STT")

        prediction = _get_text_clf().predict(final_transcript)

        spoken_response = generate_response(prediction.label)

        return AnalysisResponse(
            transcript=final_transcript,
            prediction=prediction,
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
