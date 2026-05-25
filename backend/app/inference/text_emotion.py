import logging

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline as hf_pipeline

from app.config import (
    TEXT_MODEL_DIR,
    FALLBACK_TEXT_MODEL,
    EMOTION_LABELS,
    DEVICE,
)
from app.schemas import EmotionPrediction

logger = logging.getLogger(__name__)


def _neutral_prediction(confidence: float = 0.3) -> EmotionPrediction:
    probs = {label: 0.0 for label in EMOTION_LABELS}
    probs["neutral"] = confidence
    remaining = (1.0 - confidence) / (len(EMOTION_LABELS) - 1)
    for label in EMOTION_LABELS:
        if label != "neutral":
            probs[label] = remaining
    return EmotionPrediction(label="neutral", confidence=confidence, probabilities=probs)


class TextEmotionClassifier:
    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._pipeline = None
        self._using_custom = False

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._pipeline is not None:
            return

        if TEXT_MODEL_DIR.exists() and any(TEXT_MODEL_DIR.iterdir()):
            try:
                logger.info("Loading custom text emotion model from %s", TEXT_MODEL_DIR)
                self._tokenizer = AutoTokenizer.from_pretrained(str(TEXT_MODEL_DIR))
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    str(TEXT_MODEL_DIR)
                ).to(DEVICE)
                self._model.eval()
                self._using_custom = True
                logger.info("Custom text emotion model loaded on %s", DEVICE)
                return
            except Exception as e:
                logger.warning("Failed to load custom text model: %s", e)

        try:
            logger.info("Loading fallback text model: %s", FALLBACK_TEXT_MODEL)
            self._pipeline = hf_pipeline(
                "text-classification",
                model=FALLBACK_TEXT_MODEL,
                top_k=None,
                device=DEVICE,
            )
            logger.info("Fallback text model loaded on %s", DEVICE)
        except Exception as e:
            logger.error("Failed to load fallback text model: %s", e)

    def predict(self, text: str) -> EmotionPrediction:
        if not text or len(text.strip()) < 3:
            return _neutral_prediction(0.3)

        self._ensure_loaded()

        try:
            if self._using_custom and self._model is not None:
                return self._predict_custom(text)
            if self._pipeline is not None:
                return self._predict_fallback(text)
        except Exception as e:
            logger.error("Text emotion prediction failed: %s", e)

        return _neutral_prediction(0.3)

    def _predict_custom(self, text: str) -> EmotionPrediction:
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(DEVICE)
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
        if isinstance(probs, float):
            probs = [probs]

        num_labels = len(EMOTION_LABELS)
        if len(probs) != num_labels:
            logger.warning(
                "Custom model output %d classes, expected %d — falling back",
                len(probs), num_labels,
            )
            return _neutral_prediction(0.3)

        probabilities = {
            EMOTION_LABELS[i]: float(probs[i]) for i in range(num_labels)
        }
        top_label = max(probabilities, key=probabilities.get)
        return EmotionPrediction(
            label=top_label,
            confidence=probabilities[top_label],
            probabilities=probabilities,
        )

    def get_embedding(self, text: str) -> torch.Tensor | None:
        """Extract CLS hidden state from the custom model. Returns None if using fallback."""
        self._ensure_loaded()
        if not self._using_custom or self._model is None:
            return None
        try:
            inputs = self._tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(DEVICE)
            with torch.no_grad():
                outputs = self._model(**inputs, output_hidden_states=True)
            return outputs.hidden_states[-1][:, 0, :].squeeze(0).cpu()
        except Exception as e:
            logger.warning("Failed to extract text embedding: %s", e)
            return None

    def _predict_fallback(self, text: str) -> EmotionPrediction:
        results = self._pipeline(text)[0]
        raw = {r["label"]: float(r["score"]) for r in results}

        probabilities = {label: raw.get(label, 0.0) for label in EMOTION_LABELS}

        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v / total for k, v in probabilities.items()}

        top_label = max(probabilities, key=probabilities.get)
        return EmotionPrediction(
            label=top_label,
            confidence=probabilities[top_label],
            probabilities=probabilities,
        )
