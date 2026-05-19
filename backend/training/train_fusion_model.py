"""Train a gated late-fusion model combining text and audio embeddings on MELD."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import (
    PROCESSED_DIR, DEVICE, EMOTION_LABELS, NUM_LABELS,
    TEXT_MODEL_DIR, AUDIO_MODEL_DIR, FUSION_MODEL_DIR,
)

import argparse
import json
import logging
from copy import deepcopy

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABEL_TO_ID = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
PROJ_DIM = 256
FEATURE_CACHE_DIR = PROCESSED_DIR / "meld" / "features"


class GatedFusionModel(nn.Module):
    def __init__(self, text_emb_dim=768, audio_emb_dim=768, proj_dim=PROJ_DIM,
                 num_labels=NUM_LABELS):
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

    def forward(self, text_emb, audio_emb, text_probs, audio_probs):
        gate = self.gate_net(torch.cat([text_emb, audio_emb], dim=-1))
        fused = gate * self.text_proj(text_emb) + (1 - gate) * self.audio_proj(audio_emb)
        logits = self.classifier(torch.cat([fused, text_probs, audio_probs], dim=-1))
        return logits


def load_meld_data(meld_dir: Path) -> dict[str, pd.DataFrame]:
    split_names = {"train": "train.csv", "dev": "dev.csv", "test": "test.csv"}
    dfs = {}
    for split, filename in split_names.items():
        path = meld_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run prepare_meld.py first.")
        df = pd.read_csv(path)
        df = df[df["utterance_text"].notna() & (df["audio_path"] != "")].reset_index(drop=True)
        df = df[df["audio_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
        dfs[split] = df
        logger.info(f"MELD {split}: {len(df)} examples with both text and audio")
    return dfs


@torch.no_grad()
def extract_text_features(texts: list[str], model, tokenizer, device: str, batch_size: int = 32):
    model.eval()
    all_embeddings = []
    all_probs = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting text features"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, max_length=128, padding="max_length",
            truncation=True, return_tensors="pt"
        ).to(device)

        outputs = model(**inputs, output_hidden_states=True)
        cls_hidden = outputs.hidden_states[-1][:, 0, :]
        probs = F.softmax(outputs.logits, dim=-1)

        all_embeddings.append(cls_hidden.cpu())
        all_probs.append(probs.cpu())

    return torch.cat(all_embeddings, dim=0), torch.cat(all_probs, dim=0)


@torch.no_grad()
def extract_audio_features(
    audio_paths: list[str], model, feature_extractor, device: str, batch_size: int = 8
):
    model.eval()
    all_embeddings = []
    all_probs = []
    max_samples = 16000 * 5

    for i in tqdm(range(0, len(audio_paths), batch_size), desc="Extracting audio features"):
        batch_paths = audio_paths[i:i + batch_size]
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

        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        pooled = hidden.mean(dim=1)
        probs = F.softmax(outputs.logits, dim=-1)

        all_embeddings.append(pooled.cpu())
        all_probs.append(probs.cpu())

    return torch.cat(all_embeddings, dim=0), torch.cat(all_probs, dim=0)


def get_or_extract_features(
    split_name: str, df: pd.DataFrame,
    text_model, tokenizer, audio_model, feature_extractor, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load cached features or extract them from pretrained models."""
    cache_dir = FEATURE_CACHE_DIR / split_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_files = {
        "text_emb": cache_dir / "text_embeddings.pt",
        "text_probs": cache_dir / "text_probs.pt",
        "audio_emb": cache_dir / "audio_embeddings.pt",
        "audio_probs": cache_dir / "audio_probs.pt",
        "labels": cache_dir / "labels.pt",
    }

    if all(f.exists() for f in cache_files.values()):
        logger.info(f"Loading cached features for {split_name}")
        return tuple(torch.load(f, weights_only=True) for f in cache_files.values())

    logger.info(f"Extracting features for {split_name}...")

    texts = df["utterance_text"].tolist()
    audio_paths = df["audio_path"].tolist()
    labels = torch.tensor(df["label_id"].values, dtype=torch.long)

    text_emb, text_probs = extract_text_features(texts, text_model, tokenizer, device)
    audio_emb, audio_probs = extract_audio_features(
        audio_paths, audio_model, feature_extractor, device
    )

    for name, tensor in zip(cache_files.keys(),
                            [text_emb, text_probs, audio_emb, audio_probs, labels]):
        torch.save(tensor, cache_files[name])

    logger.info(f"Cached features for {split_name} to {cache_dir}")
    return text_emb, text_probs, audio_emb, audio_probs, labels


def train_fusion(
    train_data, val_data, class_weights: np.ndarray,
    device: str, epochs: int = 20, batch_size: int = 32, lr: float = 1e-3,
    patience: int = 5,
):
    train_dataset = TensorDataset(*train_data)
    val_dataset = TensorDataset(*val_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    text_emb_dim = train_data[0].shape[1]
    audio_emb_dim = train_data[2].shape[1]

    model = GatedFusionModel(
        text_emb_dim=text_emb_dim,
        audio_emb_dim=audio_emb_dim,
        proj_dim=PROJ_DIM,
        num_labels=NUM_LABELS,
    ).to(device)

    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            text_emb, text_probs, audio_emb, audio_probs, labels = [
                b.to(device) for b in batch
            ]

            optimizer.zero_grad()
            logits = model(text_emb, audio_emb, text_probs, audio_probs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=-1) == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                text_emb, text_probs, audio_emb, audio_probs, labels = [
                    b.to(device) for b in batch
                ]
                logits = model(text_emb, audio_emb, text_probs, audio_probs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
                val_preds.extend(logits.argmax(dim=-1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        avg_train_loss = train_loss / train_total
        avg_val_loss = val_loss / len(val_dataset)
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        val_acc = accuracy_score(val_labels, val_preds)

        logger.info(
            f"Epoch {epoch+1:2d}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} Acc: {train_correct/train_total:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} (patience={patience})")
                break

    model.load_state_dict(best_state)
    return model


def evaluate_model(model, test_data, device: str):
    test_dataset = TensorDataset(*test_data)
    loader = DataLoader(test_dataset, batch_size=64)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            text_emb, text_probs, audio_emb, audio_probs, labels = [
                b.to(device) for b in batch
            ]
            logits = model(text_emb, audio_emb, text_probs, audio_probs)
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1, np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description="Train fusion emotion model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    for model_dir, name in [(TEXT_MODEL_DIR, "text"), (AUDIO_MODEL_DIR, "audio")]:
        if not model_dir.exists() or not any(model_dir.iterdir()):
            logger.error(
                f"Trained {name} model not found at {model_dir}. "
                f"Run train_{name}_model.py first."
            )
            sys.exit(1)

    meld_dir = PROCESSED_DIR / "meld"
    dfs = load_meld_data(meld_dir)

    device = DEVICE
    if device == "mps":
        try:
            torch.zeros(1, device="mps")
        except Exception:
            logger.warning("MPS not available, falling back to CPU")
            device = "cpu"

    logger.info(f"Using device: {device}")

    logger.info(f"Loading text model from {TEXT_MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(TEXT_MODEL_DIR))
    text_model = AutoModelForSequenceClassification.from_pretrained(
        str(TEXT_MODEL_DIR), output_hidden_states=True
    ).to(device)

    logger.info(f"Loading audio model from {AUDIO_MODEL_DIR}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(str(AUDIO_MODEL_DIR))
    audio_model = AutoModelForAudioClassification.from_pretrained(
        str(AUDIO_MODEL_DIR), output_hidden_states=True
    ).to(device)

    features = {}
    for split_name, df in dfs.items():
        features[split_name] = get_or_extract_features(
            split_name, df, text_model, tokenizer, audio_model, feature_extractor, device
        )

    del text_model, audio_model
    torch.cuda.empty_cache() if device == "cuda" else None

    train_labels = features["train"][4].numpy()
    class_weights = compute_class_weight(
        "balanced", classes=np.arange(NUM_LABELS), y=train_labels
    )
    logger.info(f"Class weights: {dict(zip(EMOTION_LABELS, class_weights.round(3)))}")

    logger.info("Training fusion model...")
    model = train_fusion(
        train_data=features["train"],
        val_data=features["dev"],
        class_weights=class_weights,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )

    logger.info("Evaluating on test set...")
    test_acc, test_f1, _, _ = evaluate_model(model, features["test"], device)

    FUSION_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), FUSION_MODEL_DIR / "fusion_model.pt")

    config = {
        "text_emb_dim": 768,
        "audio_emb_dim": 768,
        "proj_dim": PROJ_DIM,
        "num_labels": NUM_LABELS,
    }
    with open(FUSION_MODEL_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    label_map = {i: label for i, label in enumerate(EMOTION_LABELS)}
    with open(FUSION_MODEL_DIR / "label_mapping.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print("\n" + "=" * 60)
    print("Fusion Model Training Results")
    print("=" * 60)
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Macro F1:  {test_f1:.4f}")
    print(f"\nModel saved to: {FUSION_MODEL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
