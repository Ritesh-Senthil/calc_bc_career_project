import logging

import librosa
import numpy as np
import torch
from huggingface_hub import hf_hub_download
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

_CLASSIFIER_KEY_REMAP = {
    "classifier.dense.weight": "projector.weight",
    "classifier.dense.bias": "projector.bias",
    "classifier.output.weight": "classifier.weight",
    "classifier.output.bias": "classifier.bias",
}

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

            self._fix_fallback_classifier(model)

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

    @staticmethod
    def _fix_fallback_classifier(model: Wav2Vec2ForSequenceClassification) -> None:
        """Remap old-format checkpoint keys to current transformers layout.

        The ehcalabres checkpoint stores the classification head as
        classifier.dense.* / classifier.output.*, but current transformers
        expects projector.* / classifier.*.  Without remapping, those layers
        stay randomly initialized and the model predicts near-uniform probs.
        """
        try:
            weights_path = hf_hub_download(
                FALLBACK_AUDIO_MODEL, "model.safetensors"
            )
            from safetensors.torch import load_file
            raw = load_file(weights_path)
        except Exception:
            try:
                weights_path = hf_hub_download(
                    FALLBACK_AUDIO_MODEL, "pytorch_model.bin"
                )
                raw = torch.load(weights_path, map_location="cpu", weights_only=True)
            except Exception as e:
                logger.warning("Could not reload checkpoint for key remapping: %s", e)
                return

        state = model.state_dict()
        patched = 0
        for old_key, new_key in _CLASSIFIER_KEY_REMAP.items():
            if old_key in raw and new_key in state:
                if raw[old_key].shape == state[new_key].shape:
                    state[new_key] = raw[old_key]
                    patched += 1

        if patched:
            model.load_state_dict(state)
            logger.info("Patched %d classifier keys from fallback checkpoint", patched)
        else:
            logger.warning("No classifier keys matched for remapping")

    def _get_model_labels(self) -> list[str] | None:
        """Read label ordering from the model's own config instead of guessing."""
        try:
            id2label = self._model.config.id2label
            if id2label:
                return [id2label[i] for i in range(len(id2label))]
        except Exception:
            pass
        return None

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

            model_labels = self._get_model_labels()
            return self._build_fallback_prediction(raw_probs, model_labels)

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

    def _build_fallback_prediction(
        self, raw_probs: np.ndarray, model_labels: list[str] | None = None,
    ) -> EmotionPrediction:
        if model_labels is None:
            model_labels = [
                "angry", "calm", "disgust", "fearful",
                "happy", "neutral", "sad", "surprised",
            ]

        raw_map = {
            model_labels[i]: float(raw_probs[i])
            for i in range(min(len(model_labels), len(raw_probs)))
        }

        logger.info("Audio raw label probs: %s",
                     {k: f"{v:.3f}" for k, v in raw_map.items()})

        probabilities: dict[str, float] = {lbl: 0.0 for lbl in EMOTION_LABELS}

        for src_label, prob in raw_map.items():
            src_lower = src_label.lower()
            if src_lower not in _FALLBACK_LABEL_MAP:
                continue
            target = _FALLBACK_LABEL_MAP[src_lower]

            if src_lower == "calm":
                # Calm ≠ neutral — redistribute: 40% to neutral, 60% spread
                # across other emotions proportionally.  Straight summing
                # gives neutral a ~2x structural advantage.
                probabilities["neutral"] += prob * 0.4
                spread = prob * 0.6 / max(len(EMOTION_LABELS) - 1, 1)
                for lbl in EMOTION_LABELS:
                    if lbl != "neutral":
                        probabilities[lbl] += spread
            else:
                probabilities[target] = probabilities.get(target, 0.0) + prob

        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v / total for k, v in probabilities.items()}

        top_label = max(probabilities, key=probabilities.get)
        return EmotionPrediction(
            label=top_label,
            confidence=probabilities[top_label],
            probabilities=probabilities,
        )
