"""Fine-tune DeBERTa-v3-base for 7-class text emotion classification."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import (
    PROCESSED_DIR, MODELS_DIR, DEVICE, EMOTION_LABELS, NUM_LABELS,
    TEXT_MODEL_DIR, TEXT_BASE_MODEL, TRAINING_BATCH_SIZE, TRAINING_EPOCHS,
    LEARNING_RATE,
)

import argparse
import json
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import evaluate as hf_evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABEL_TO_ID = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
ID_TO_LABEL = {idx: label for idx, label in enumerate(EMOTION_LABELS)}


class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class WeightedTrainer(Trainer):
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


def load_data(data_dir: Path):
    dfs = {}
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run prepare_goemotions.py first."
            )
        dfs[split] = pd.read_csv(path)
        logger.info(f"Loaded {split}: {len(dfs[split])} examples")
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
    parser = argparse.ArgumentParser(description="Train text emotion model")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    epochs = args.epochs or TRAINING_EPOCHS
    batch_size = args.batch_size or TRAINING_BATCH_SIZE
    lr = args.lr or LEARNING_RATE

    data_dir = PROCESSED_DIR / "goemotions"
    dfs = load_data(data_dir)

    logger.info(f"Loading tokenizer: {TEXT_BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_BASE_MODEL)

    logger.info("Tokenizing datasets...")
    encodings = {}
    labels = {}
    for split in ("train", "val", "test"):
        texts = dfs[split]["text"].tolist()
        encodings[split] = tokenizer(
            texts, max_length=128, padding="max_length", truncation=True,
            return_tensors="pt",
        )
        labels[split] = dfs[split]["label_id"].tolist()

    datasets = {
        split: EmotionDataset(encodings[split], labels[split])
        for split in ("train", "val", "test")
    }

    train_labels = np.array(labels["train"])
    class_weights = compute_class_weight(
        "balanced", classes=np.arange(NUM_LABELS), y=train_labels
    )
    logger.info(f"Class weights: {dict(zip(EMOTION_LABELS, class_weights.round(3)))}")

    logger.info(f"Loading model: {TEXT_BASE_MODEL}")
    model = AutoModelForSequenceClassification.from_pretrained(
        TEXT_BASE_MODEL,
        num_labels=NUM_LABELS,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    TEXT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(TEXT_MODEL_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=False,
        use_mps_device=(DEVICE == "mps"),
        logging_steps=50,
        warmup_ratio=0.1,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        compute_metrics=compute_metrics_fn,
    )

    logger.info(f"Training on {DEVICE} for {epochs} epochs, batch_size={batch_size}, lr={lr}")
    trainer.train()

    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(datasets["test"], metric_key_prefix="test")

    logger.info(f"Saving model to {TEXT_MODEL_DIR}")
    trainer.save_model(str(TEXT_MODEL_DIR))
    tokenizer.save_pretrained(str(TEXT_MODEL_DIR))

    label_map = {i: label for i, label in enumerate(EMOTION_LABELS)}
    label_map_path = TEXT_MODEL_DIR / "label_mapping.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)

    print("\n" + "=" * 60)
    print("Text Model Training Results")
    print("=" * 60)
    print(f"Model: {TEXT_BASE_MODEL}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"\nTest Results:")
    for key, value in sorted(test_results.items()):
        if isinstance(value, float):
            print(f"  {key:25s}  {value:.4f}")
    print(f"\nModel saved to: {TEXT_MODEL_DIR}")
    print(f"Label mapping: {label_map_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
