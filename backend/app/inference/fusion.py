import json
import logging

import torch
import torch.nn as nn

from app.config import FUSION_MODEL_DIR, DEFAULT_TEXT_WEIGHT, EMOTION_LABELS, NUM_LABELS
from app.schemas import EmotionPrediction

logger = logging.getLogger(__name__)


class GatedFusionModel(nn.Module):
    """Architecture must match training/train_fusion_model.py exactly."""

    def __init__(self, text_emb_dim: int = 768, audio_emb_dim: int = 768,
                 proj_dim: int = 256, num_labels: int = NUM_LABELS):
        super().__init__()
        self.text_proj = nn.Linear(text_emb_dim, proj_dim)
        self.audio_proj = nn.Linear(audio_emb_dim, proj_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(text_emb_dim + audio_emb_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim + num_labels + num_labels, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_labels),
        )

    def forward(self, text_emb: torch.Tensor, audio_emb: torch.Tensor,
                text_probs: torch.Tensor, audio_probs: torch.Tensor) -> torch.Tensor:
        gate = self.gate_net(torch.cat([text_emb, audio_emb], dim=-1))
        fused = gate * self.text_proj(text_emb) + (1 - gate) * self.audio_proj(audio_emb)
        return self.classifier(torch.cat([fused, text_probs, audio_probs], dim=-1))


class FusionPredictor:
    def __init__(self) -> None:
        self._model: GatedFusionModel | None = None
        self._using_trained = False
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        weights_path = FUSION_MODEL_DIR / "fusion_model.pt"
        config_path = FUSION_MODEL_DIR / "config.json"

        if weights_path.exists() and config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                model = GatedFusionModel(
                    text_emb_dim=config.get("text_emb_dim", 768),
                    audio_emb_dim=config.get("audio_emb_dim", 768),
                    proj_dim=config.get("proj_dim", 256),
                    num_labels=config.get("num_labels", NUM_LABELS),
                )
                model.load_state_dict(torch.load(weights_path, map_location="cpu"))
                model.eval()
                self._model = model
                self._using_trained = True
                logger.info("Loaded trained fusion model")
                return
            except Exception as e:
                logger.warning("Failed to load trained fusion model: %s", e)

        logger.info("Using heuristic fusion")

    def predict(
        self,
        text_prediction: EmotionPrediction,
        audio_prediction: EmotionPrediction,
        transcript: str,
        text_embedding: torch.Tensor | None = None,
        audio_embedding: torch.Tensor | None = None,
    ) -> EmotionPrediction:
        self._ensure_loaded()

        try:
            if (self._using_trained and self._model is not None
                    and text_embedding is not None and audio_embedding is not None):
                return self._predict_trained(
                    text_prediction, audio_prediction, text_embedding, audio_embedding
                )
            return self._predict_heuristic(text_prediction, audio_prediction, transcript)
        except Exception as e:
            logger.error("Fusion prediction failed, using simple average: %s", e)
            return self._simple_average(text_prediction, audio_prediction)

    def _predict_trained(
        self,
        text_pred: EmotionPrediction,
        audio_pred: EmotionPrediction,
        text_emb: torch.Tensor,
        audio_emb: torch.Tensor,
    ) -> EmotionPrediction:
        text_probs = torch.tensor(
            [text_pred.probabilities.get(lbl, 0.0) for lbl in EMOTION_LABELS]
        ).unsqueeze(0)
        audio_probs = torch.tensor(
            [audio_pred.probabilities.get(lbl, 0.0) for lbl in EMOTION_LABELS]
        ).unsqueeze(0)
        text_emb = text_emb.unsqueeze(0) if text_emb.dim() == 1 else text_emb
        audio_emb = audio_emb.unsqueeze(0) if audio_emb.dim() == 1 else audio_emb

        with torch.no_grad():
            logits = self._model(text_emb, audio_emb, text_probs, audio_probs)
            fused_probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()

        probabilities = {EMOTION_LABELS[i]: fused_probs[i] for i in range(len(EMOTION_LABELS))}
        top_label = max(probabilities, key=probabilities.get)
        return EmotionPrediction(
            label=top_label,
            confidence=probabilities[top_label],
            probabilities=probabilities,
        )

    def _predict_heuristic(
        self,
        text_pred: EmotionPrediction,
        audio_pred: EmotionPrediction,
        transcript: str,
    ) -> EmotionPrediction:
        gate = DEFAULT_TEXT_WEIGHT

        if text_pred.confidence > audio_pred.confidence + 0.2:
            gate = 0.75
        elif audio_pred.confidence > text_pred.confidence + 0.2:
            gate = 0.35

        if len(transcript.split()) < 5:
            gate = 0.3

        probabilities: dict[str, float] = {}
        for label in EMOTION_LABELS:
            t = text_pred.probabilities.get(label, 0.0)
            a = audio_pred.probabilities.get(label, 0.0)
            probabilities[label] = gate * t + (1 - gate) * a

        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v / total for k, v in probabilities.items()}

        top_label = max(probabilities, key=probabilities.get)
        return EmotionPrediction(
            label=top_label,
            confidence=probabilities[top_label],
            probabilities=probabilities,
        )

    @staticmethod
    def _simple_average(
        text_pred: EmotionPrediction,
        audio_pred: EmotionPrediction,
    ) -> EmotionPrediction:
        probabilities: dict[str, float] = {}
        for label in EMOTION_LABELS:
            t = text_pred.probabilities.get(label, 0.0)
            a = audio_pred.probabilities.get(label, 0.0)
            probabilities[label] = (t + a) / 2.0

        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v / total for k, v in probabilities.items()}

        top_label = max(probabilities, key=probabilities.get)
        return EmotionPrediction(
            label=top_label,
            confidence=probabilities[top_label],
            probabilities=probabilities,
        )
