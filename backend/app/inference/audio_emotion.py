import logging

import librosa
import numpy as np
import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)

from app.config import (
    AUDIO_MODEL_DIR,
    FALLBACK_AUDIO_MODEL,
    EMOTION_LABELS,
    SAMPLE_RATE,
    DEVICE,
)
from app.schemas import EmotionPrediction

logger = logging.getLogger(__name__)

# Fallback model uses 8 labels that need remapping to our 7
_FALLBACK_LABEL_MAP = {
    "angry": "anger",
    "calm": "neutral",
    "disgust": "disgust",
    "fearful": "fear",
    "happy": "joy",
    "neutral": "neutral",
    "sad": "sadness",
    "surprised": "surprise",
}


def _neutral_prediction(confidence: float = 0.3) -> EmotionPrediction:
    probs = {label: 0.0 for label in EMOTION_LABELS}
    probs["neutral"] = confidence
    remaining = (1.0 - confidence) / (len(EMOTION_LABELS) - 1)
    for label in EMOTION_LABELS:
        if label != "neutral":
            probs[label] = remaining
    return EmotionPrediction(label="neutral", confidence=confidence, probabilities=probs)


class AudioEmotionClassifier:
    def __init__(self) -> None:
        self._feature_extractor = None
        self._model = None
        self._using_custom = False
        self._device: str = DEVICE

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        if AUDIO_MODEL_DIR.exists() and any(AUDIO_MODEL_DIR.iterdir()):
            try:
                logger.info("Loading custom audio model from %s", AUDIO_MODEL_DIR)
                self._feature_extractor = AutoFeatureExtractor.from_pretrained(
                    str(AUDIO_MODEL_DIR)
                )
                self._model = AutoModelForAudioClassification.from_pretrained(
                    str(AUDIO_MODEL_DIR)
                ).to(self._device)
                self._model.eval()
                self._using_custom = True
                logger.info("Custom audio model loaded on %s", self._device)
                return
            except Exception as e:
                logger.warning("Failed to load custom audio model: %s", e)

        try:
            logger.info("Loading fallback audio model: %s", FALLBACK_AUDIO_MODEL)
            self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                FALLBACK_AUDIO_MODEL
            )
            model = Wav2Vec2ForSequenceClassification.from_pretrained(
                FALLBACK_AUDIO_MODEL
            )
            try:
                self._model = model.to(self._device)
            except RuntimeError:
                logger.warning("MPS failed for audio model, falling back to CPU")
                self._device = "cpu"
                self._model = model.to("cpu")
            self._model.eval()
            self._using_custom = False
            logger.info("Fallback audio model loaded on %s", self._device)
        except Exception as e:
            logger.error("Failed to load fallback audio model: %s", e)

    def predict(self, audio_path: str) -> EmotionPrediction:
        self._ensure_loaded()

        if self._model is None:
            return _neutral_prediction(0.3)

        try:
            audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            inputs = self._feature_extractor(
                audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self._model(**inputs).logits
            raw_probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

            if self._using_custom:
                return self._build_custom_prediction(raw_probs)
            return self._build_fallback_prediction(raw_probs)

        except Exception as e:
            logger.error("Audio emotion prediction failed: %s", e)
            return _neutral_prediction(0.3)

    def get_embedding(self, audio_path: str) -> torch.Tensor | None:
        """Extract mean-pooled hidden state from the custom model. Returns None if using fallback."""
        self._ensure_loaded()
        if not self._using_custom or self._model is None:
            return None
        try:
            audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            inputs = self._feature_extractor(
                audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs, output_hidden_states=True)
            return outputs.hidden_states[-1].mean(dim=1).squeeze(0).cpu()
        except Exception as e:
            logger.warning("Failed to extract audio embedding: %s", e)
            return None

    def _build_custom_prediction(self, raw_probs: np.ndarray) -> EmotionPrediction:
        probabilities = {
            EMOTION_LABELS[i]: float(raw_probs[i]) for i in range(len(EMOTION_LABELS))
        }
        top_label = max(probabilities, key=probabilities.get)
        return EmotionPrediction(
            label=top_label,
            confidence=probabilities[top_label],
            probabilities=probabilities,
        )

    def _build_fallback_prediction(self, raw_probs: np.ndarray) -> EmotionPrediction:
        fallback_labels = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
        raw_map = {
            fallback_labels[i]: float(raw_probs[i])
            for i in range(len(fallback_labels))
        }

        # "calm" and "neutral" both map to our "neutral" — sum their probabilities
        probabilities: dict[str, float] = {}
        for src_label, prob in raw_map.items():
            target = _FALLBACK_LABEL_MAP[src_label]
            probabilities[target] = probabilities.get(target, 0.0) + prob

        for label in EMOTION_LABELS:
            probabilities.setdefault(label, 0.0)

        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v / total for k, v in probabilities.items()}

        top_label = max(probabilities, key=probabilities.get)
        return EmotionPrediction(
            label=top_label,
            confidence=probabilities[top_label],
            probabilities=probabilities,
        )
