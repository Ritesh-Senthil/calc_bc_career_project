"""Evaluate MoodMirror emotion models with metrics and confusion matrices."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.config import (
    PROCESSED_DIR, MODELS_DIR, DEVICE, EMOTION_LABELS, NUM_LABELS,
    TEXT_MODEL_DIR, AUDIO_MODEL_DIR, FUSION_MODEL_DIR,
    TEXT_BASE_MODEL, AUDIO_BASE_MODEL,
)

import argparse
import json
import logging

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from tqdm import tqdm
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABEL_TO_ID = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
ID_TO_LABEL = {idx: label for idx, label in enumerate(EMOTION_LABELS)}


def save_confusion_matrix(y_true, y_pred, labels, save_path: Path, title: str):
    plt.style.use("dark_background")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="magma",
        xticklabels=labels, yticklabels=labels, ax=ax,
        linewidths=0.5, linecolor="#333333",
        cbar_kws={"label": "Proportion"},
    )

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j + 0.5, i + 0.72, f"({cm[i, j]})",
                ha="center", va="center", fontsize=7, color="#aaaaaa",
            )

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14, pad=15)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {save_path}")


def evaluate_text_model():
    logger.info("=== Evaluating Text Model ===")

    test_path = PROCESSED_DIR / "goemotions" / "test.csv"
    if not test_path.exists():
        logger.error(f"Test data not found: {test_path}")
        return None

    df = pd.read_csv(test_path)
    logger.info(f"Loaded {len(df)} test examples")

    model_dir = TEXT_MODEL_DIR
    base_model = TEXT_BASE_MODEL

    if model_dir.exists() and (model_dir / "config.json").exists():
        logger.info(f"Loading trained model from {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    else:
        logger.warning(f"Trained model not found at {model_dir}, using base: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=NUM_LABELS
        )

    device = DEVICE
    model = model.to(device).eval()

    all_preds = []
    batch_size = 32
    texts = df["text"].tolist()

    for i in tqdm(range(0, len(texts), batch_size), desc="Text predictions"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, max_length=128, padding="max_length",
            truncation=True, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)

    true_labels = df["label_id"].values
    pred_labels = np.array(all_preds)

    acc = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average="macro")
    f1_weighted = f1_score(true_labels, pred_labels, average="weighted")

    report = classification_report(
        true_labels, pred_labels, target_names=EMOTION_LABELS, digits=4
    )

    save_confusion_matrix(
        true_labels, pred_labels, EMOTION_LABELS,
        TEXT_MODEL_DIR / "confusion_matrix.png",
        "Text Model Confusion Matrix (GoEmotions)",
    )

    return {
        "model": "Text (DeBERTa-v3)",
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "report": report,
    }


def evaluate_audio_model():
    logger.info("=== Evaluating Audio Model ===")

    test_path = PROCESSED_DIR / "audio" / "test.csv"
    if not test_path.exists():
        logger.error(f"Test data not found: {test_path}")
        return None

    df = pd.read_csv(test_path)
    df = df[df["file_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    logger.info(f"Loaded {len(df)} test examples (with valid audio paths)")

    if len(df) == 0:
        logger.error("No valid audio files found in test set")
        return None

    model_dir = AUDIO_MODEL_DIR
    base_model = AUDIO_BASE_MODEL

    if model_dir.exists() and (model_dir / "config.json").exists():
        logger.info(f"Loading trained model from {model_dir}")
        feature_extractor = AutoFeatureExtractor.from_pretrained(str(model_dir))
        model = AutoModelForAudioClassification.from_pretrained(str(model_dir))
    else:
        logger.warning(f"Trained model not found at {model_dir}, using base: {base_model}")
        feature_extractor = AutoFeatureExtractor.from_pretrained(base_model)
        model = AutoModelForAudioClassification.from_pretrained(
            base_model, num_labels=NUM_LABELS
        )

    device = DEVICE
    model = model.to(device).eval()

    max_samples = 16000 * 5
    all_preds = []
    batch_size = 8

    file_paths = df["file_path"].tolist()
    for i in tqdm(range(0, len(file_paths), batch_size), desc="Audio predictions"):
        batch_paths = file_paths[i:i + batch_size]
        waveforms = []

        for path in batch_paths:
            try:
                audio, _ = librosa.load(path, sr=16000, mono=True)
            except Exception:
                audio = np.zeros(max_samples, dtype=np.float32)

            if len(audio) > max_samples:
                audio = audio[:max_samples]
            elif len(audio) < max_samples:
                audio = np.pad(audio, (0, max_samples - len(audio)))
            waveforms.append(audio)

        inputs = feature_extractor(
            waveforms, sampling_rate=16000, return_tensors="pt", padding=True
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)

    true_labels = df["label_id"].values
    pred_labels = np.array(all_preds)

    acc = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average="macro")
    f1_weighted = f1_score(true_labels, pred_labels, average="weighted")

    report = classification_report(
        true_labels, pred_labels, target_names=EMOTION_LABELS, digits=4
    )

    save_confusion_matrix(
        true_labels, pred_labels, EMOTION_LABELS,
        AUDIO_MODEL_DIR / "confusion_matrix.png",
        "Audio Model Confusion Matrix (RAVDESS + CREMA-D)",
    )

    return {
        "model": "Audio (Wav2Vec2)",
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "report": report,
    }


def evaluate_fusion_model():
    logger.info("=== Evaluating Fusion Model ===")

    test_path = PROCESSED_DIR / "meld" / "test.csv"
    if not test_path.exists():
        logger.error(f"Test data not found: {test_path}")
        return None

    fusion_weights = FUSION_MODEL_DIR / "fusion_model.pt"
    fusion_config = FUSION_MODEL_DIR / "config.json"
    if not fusion_weights.exists():
        logger.error(f"Fusion model not found at {fusion_weights}")
        return None

    df = pd.read_csv(test_path)
    df = df[
        df["utterance_text"].notna()
        & (df["audio_path"] != "")
        & df["audio_path"].apply(lambda p: Path(p).exists())
    ].reset_index(drop=True)
    logger.info(f"Loaded {len(df)} test examples (with text and audio)")

    if len(df) == 0:
        logger.error("No valid test examples with both text and audio")
        return None

    device = DEVICE

    logger.info("Loading pretrained text model for feature extraction...")
    tokenizer = AutoTokenizer.from_pretrained(str(TEXT_MODEL_DIR))
    text_model = AutoModelForSequenceClassification.from_pretrained(
        str(TEXT_MODEL_DIR), output_hidden_states=True
    ).to(device).eval()

    logger.info("Loading pretrained audio model for feature extraction...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(str(AUDIO_MODEL_DIR))
    audio_model = AutoModelForAudioClassification.from_pretrained(
        str(AUDIO_MODEL_DIR), output_hidden_states=True
    ).to(device).eval()

    cache_dir = PROCESSED_DIR / "meld" / "features" / "test"
    cache_files = {
        "text_emb": cache_dir / "text_embeddings.pt",
        "text_probs": cache_dir / "text_probs.pt",
        "audio_emb": cache_dir / "audio_embeddings.pt",
        "audio_probs": cache_dir / "audio_probs.pt",
    }

    if all(f.exists() for f in cache_files.values()):
        logger.info("Loading cached test features")
        text_emb = torch.load(cache_files["text_emb"], weights_only=True)
        text_probs = torch.load(cache_files["text_probs"], weights_only=True)
        audio_emb = torch.load(cache_files["audio_emb"], weights_only=True)
        audio_probs = torch.load(cache_files["audio_probs"], weights_only=True)
    else:
        logger.info("Extracting test features (this may take a while)...")
        from train_fusion_model import extract_text_features, extract_audio_features

        text_emb, text_probs = extract_text_features(
            df["utterance_text"].tolist(), text_model, tokenizer, device
        )
        audio_emb, audio_probs = extract_audio_features(
            df["audio_path"].tolist(), audio_model, feature_extractor, device
        )

    del text_model, audio_model

    with open(fusion_config) as f:
        config = json.load(f)

    from train_fusion_model import GatedFusionModel

    fusion_model = GatedFusionModel(
        text_emb_dim=config["text_emb_dim"],
        audio_emb_dim=config["audio_emb_dim"],
        proj_dim=config["proj_dim"],
        num_labels=config["num_labels"],
    )
    fusion_model.load_state_dict(torch.load(fusion_weights, weights_only=True, map_location=device))
    fusion_model = fusion_model.to(device).eval()

    batch_size = 64
    all_preds = []
    n = len(df)

    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        with torch.no_grad():
            logits = fusion_model(
                text_emb[i:end].to(device),
                audio_emb[i:end].to(device),
                text_probs[i:end].to(device),
                audio_probs[i:end].to(device),
            )
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())

    true_labels = df["label_id"].values
    pred_labels = np.array(all_preds)

    acc = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average="macro")
    f1_weighted = f1_score(true_labels, pred_labels, average="weighted")

    report = classification_report(
        true_labels, pred_labels, target_names=EMOTION_LABELS, digits=4
    )

    save_confusion_matrix(
        true_labels, pred_labels, EMOTION_LABELS,
        FUSION_MODEL_DIR / "confusion_matrix.png",
        "Fusion Model Confusion Matrix (MELD)",
    )

    return {
        "model": "Fusion (Gated)",
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "report": report,
    }


def print_summary(results: list[dict]):
    print("\n" + "=" * 70)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 70)

    header = f"{'Model':25s} {'Accuracy':>10s} {'Macro F1':>10s} {'Weighted F1':>12s}"
    print(header)
    print("-" * 70)

    for r in results:
        print(
            f"{r['model']:25s} {r['accuracy']:10.4f} {r['f1_macro']:10.4f} "
            f"{r['f1_weighted']:12.4f}"
        )

    print("=" * 70)

    for r in results:
        print(f"\n--- {r['model']} Per-Class Report ---")
        print(r["report"])


def main():
    parser = argparse.ArgumentParser(description="Evaluate MoodMirror emotion models")
    parser.add_argument(
        "--model",
        choices=["all", "text", "audio", "fusion"],
        default="all",
        help="Which model to evaluate (default: all)",
    )
    args = parser.parse_args()

    evaluators = {
        "text": evaluate_text_model,
        "audio": evaluate_audio_model,
        "fusion": evaluate_fusion_model,
    }

    targets = evaluators if args.model == "all" else {args.model: evaluators[args.model]}

    results = []
    for name, func in targets.items():
        try:
            result = func()
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"Failed to evaluate {name} model: {e}")

    if results:
        print_summary(results)
    else:
        print("\nNo models could be evaluated. Ensure models are trained and data is prepared.")


if __name__ == "__main__":
    main()
