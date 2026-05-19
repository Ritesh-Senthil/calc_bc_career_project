"""Fine-tune Wav2Vec2-base for 7-class audio emotion classification."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import (
    PROCESSED_DIR, DEVICE, EMOTION_LABELS, NUM_LABELS,
    AUDIO_MODEL_DIR, AUDIO_BASE_MODEL, TRAINING_EPOCHS, LEARNING_RATE,
)

import argparse
import json
import logging

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Trainer,
    TrainingArguments,
)
import evaluate as hf_evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABEL_TO_ID = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
ID_TO_LABEL = {idx: label for idx, label in enumerate(EMOTION_LABELS)}

MAX_AUDIO_SECONDS = 5
MAX_AUDIO_SAMPLES = 16000 * MAX_AUDIO_SECONDS


class AudioEmotionDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, feature_extractor, max_samples: int = MAX_AUDIO_SAMPLES):
        self.df = df.reset_index(drop=True)
        self.feature_extractor = feature_extractor
        self.max_samples = max_samples
        self._validate_paths()

    def _validate_paths(self):
        valid_mask = self.df["file_path"].apply(lambda p: Path(p).exists())
        missing = (~valid_mask).sum()
        if missing > 0:
            logger.warning(f"Dropping {missing} samples with missing audio files")
            self.df = self.df[valid_mask].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio, _ = librosa.load(row["file_path"], sr=16000, mono=True)

        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        elif len(audio) < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))

        inputs = self.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt", padding=False
        )
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["labels"] = torch.tensor(row["label_id"], dtype=torch.long)
        return item


class WeightedAudioTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss = nn.CrossEntropyLoss(weight=weight)(logits, labels)
        else:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return (loss, outputs) if return_outputs else loss


def load_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    dfs = {}
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run prepare_audio_datasets.py first."
            )
        dfs[split] = pd.read_csv(path)
        logger.info(f"Loaded {split}: {len(dfs[split])} samples")
    return dfs


def compute_metrics_fn(eval_pred):
    accuracy_metric = hf_evaluate.load("accuracy")
    f1_metric = hf_evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1_macro = f1_metric.compute(
        predictions=predictions, references=labels, average="macro"
    )
    f1_weighted = f1_metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )

    return {
        "accuracy": accuracy["accuracy"],
        "f1_macro": f1_macro["f1"],
        "f1_weighted": f1_weighted["f1"],
    }


def main():
    parser = argparse.ArgumentParser(description="Train audio emotion model")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    epochs = args.epochs or TRAINING_EPOCHS
    batch_size = args.batch_size or 8
    lr = args.lr or LEARNING_RATE

    data_dir = PROCESSED_DIR / "audio"
    dfs = load_data(data_dir)

    logger.info(f"Loading feature extractor: {AUDIO_BASE_MODEL}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(AUDIO_BASE_MODEL)

    logger.info("Building datasets (validating audio paths)...")
    datasets = {
        split: AudioEmotionDataset(dfs[split], feature_extractor)
        for split in ("train", "val", "test")
    }

    train_labels = np.array(datasets["train"].df["label_id"].tolist())
    class_weights = compute_class_weight(
        "balanced", classes=np.arange(NUM_LABELS), y=train_labels
    )
    logger.info(f"Class weights: {dict(zip(EMOTION_LABELS, class_weights.round(3)))}")

    logger.info(f"Loading model: {AUDIO_BASE_MODEL}")
    model = AutoModelForAudioClassification.from_pretrained(
        AUDIO_BASE_MODEL,
        num_labels=NUM_LABELS,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    AUDIO_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = DEVICE
    logger.info(f"Target device: {device}")

    training_args = TrainingArguments(
        output_dir=str(AUDIO_MODEL_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=False,
        use_mps_device=(device == "mps"),
        logging_steps=50,
        warmup_ratio=0.1,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = WeightedAudioTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        compute_metrics=compute_metrics_fn,
    )

    logger.info(f"Training on {device} for {epochs} epochs, batch_size={batch_size}, lr={lr}")
    try:
        trainer.train()
    except RuntimeError as e:
        if "mps" in str(e).lower() or "MPS" in str(e):
            logger.warning(
                f"MPS training failed: {e}\n"
                "Falling back to CPU. This will be significantly slower."
            )
            training_args.use_mps_device = False
            training_args.no_cuda = True

            model = model.cpu()
            trainer = WeightedAudioTrainer(
                class_weights=class_weights,
                model=model,
                args=training_args,
                train_dataset=datasets["train"],
                eval_dataset=datasets["val"],
                compute_metrics=compute_metrics_fn,
            )
            trainer.train()
        else:
            raise

    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(datasets["test"], metric_key_prefix="test")

    test_preds = trainer.predict(datasets["test"])
    pred_labels = np.argmax(test_preds.predictions, axis=-1)
    true_labels = test_preds.label_ids

    logger.info(f"Saving model to {AUDIO_MODEL_DIR}")
    trainer.save_model(str(AUDIO_MODEL_DIR))
    feature_extractor.save_pretrained(str(AUDIO_MODEL_DIR))

    label_map = {i: label for i, label in enumerate(EMOTION_LABELS)}
    label_map_path = AUDIO_MODEL_DIR / "label_mapping.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)

    print("\n" + "=" * 60)
    print("Audio Model Training Results")
    print("=" * 60)
    print(f"Model: {AUDIO_BASE_MODEL}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"\nTest Results:")
    for key, value in sorted(test_results.items()):
        if isinstance(value, float):
            print(f"  {key:25s}  {value:.4f}")

    print(f"\nPer-class Classification Report:")
    print(classification_report(
        true_labels, pred_labels, target_names=EMOTION_LABELS, digits=4
    ))

    print(f"Model saved to: {AUDIO_MODEL_DIR}")
    print(f"Label mapping: {label_map_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
