# MoodMirror

A multimodal emotion recognition web app that analyzes both **what you say** and **how you say it**. Speak into your microphone, and MoodMirror transcribes your speech, predicts emotion from the text and vocal tone independently, fuses both signals, and speaks back a response.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Frontend (React)                  │
│                                                     │
│  Microphone → MediaRecorder → audio blob            │
│                    │                                 │
│                    ▼                                 │
│           POST /api/analyze                          │
│                    │                                 │
│                    ▼                                 │
│  ┌─────────────────────────────────────────────┐    │
│  │ Transcript │ Text Pred │ Audio Pred │ Fused  │    │
│  └─────────────────────────────────────────────┘    │
│  Probability bars + SpeechSynthesis TTS              │
└─────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│                  Backend (FastAPI)                    │
│                                                     │
│  Audio upload                                        │
│    → Convert to 16kHz mono WAV                       │
│    → STT (faster-whisper small, INT8)                │
│    → Text emotion (DeBERTa-v3-base / fallback)       │
│    → Audio emotion (Wav2Vec2-base / fallback)        │
│    → Gated late fusion (trained MLP / heuristic)     │
│    → Response generation                             │
│    → JSON response                                   │
└─────────────────────────────────────────────────────┘
```

### Emotion Classes

| Class | Description |
|-------|-------------|
| anger | Frustration, annoyance, hostility |
| disgust | Revulsion, strong disapproval |
| fear | Nervousness, anxiety, dread |
| joy | Happiness, amusement, excitement |
| neutral | Baseline, no strong emotion |
| sadness | Disappointment, grief, melancholy |
| surprise | Astonishment, confusion, curiosity |

### Models

| Component | Model | Purpose |
|-----------|-------|---------|
| STT | faster-whisper `small` | Transcription (CPU, INT8 quantized) |
| Text emotion | DeBERTa-v3-base | Fine-tuned on GoEmotions (7-class) |
| Audio emotion | Wav2Vec2-base | Fine-tuned on RAVDESS + CREMA-D |
| Fusion | Gated MLP | Trained on MELD with text + audio embeddings |

All models fall back to public HuggingFace checkpoints when custom trained weights are unavailable.

## Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3) or Linux/Windows with CUDA
- **Python 3.10+**
- **Node.js 18+**
- **ffmpeg** (required for audio conversion)

Install ffmpeg via Homebrew if not already present:

```bash
brew install ffmpeg
```

## Setup

### Backend

```bash
cd moodmirror/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the API server:

```bash
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd moodmirror/frontend
npm install
npm run dev
```

The frontend dev server starts at `http://localhost:5173` and proxies `/api` requests to the backend at `localhost:8000`.

## Training (Optional)

Training produces custom models that outperform the fallback checkpoints. The app works without training using public pretrained models.

### 1. Download Datasets

```bash
cd moodmirror/backend
python training/download_datasets.py --dataset all
```

This downloads:
- **GoEmotions** — via HuggingFace Datasets (automatic)
- **RAVDESS** — from Zenodo (~200 MB)
- **CREMA-D** — may require manual download (see script output for instructions)
- **MELD** — CSVs from GitHub, video files may require manual download

### 2. Prepare Data

```bash
python training/prepare_goemotions.py
python training/prepare_audio_datasets.py
python training/prepare_meld.py
```

### 3. Train Models

Train in this order (fusion depends on the other two):

```bash
# Text emotion classifier (~30 min on M3 Pro)
python training/train_text_model.py

# Audio emotion classifier (~45 min on M3 Pro)
python training/train_audio_model.py

# Fusion model (requires trained text + audio models)
python training/train_fusion_model.py
```

Override training hyperparameters:

```bash
python training/train_text_model.py --epochs 10 --batch-size 8 --lr 1e-5
```

### 4. Evaluate

```bash
python training/evaluate.py --model all
```

Generates accuracy, macro F1, weighted F1, and confusion matrix plots saved to each model directory.

## Dataset Details

| Dataset | Purpose | Labels | Source |
|---------|---------|--------|--------|
| GoEmotions | Text emotion training | 28 fine-grained → 7 broad | HuggingFace `go_emotions` |
| RAVDESS | Audio emotion training | 8 acted emotions | [Zenodo](https://zenodo.org/record/1188976) |
| CREMA-D | Audio emotion training | 6 acted emotions (no surprise) | [GitHub](https://github.com/CheyneyComputerScience/CREMA-D) |
| MELD | Fusion training | 7 emotions from TV dialogue | [GitHub](https://github.com/declare-lab/MELD) |

### Label Mapping

**GoEmotions → 7 classes:**
- anger ← anger, annoyance, disapproval
- disgust ← disgust
- fear ← fear, nervousness
- joy ← admiration, amusement, approval, caring, excitement, gratitude, joy, love, optimism, pride, relief
- sadness ← disappointment, embarrassment, grief, remorse, sadness
- surprise ← confusion, curiosity, realization, surprise
- neutral ← neutral

Multi-label examples that span multiple broad classes are dropped.

**RAVDESS:** calm is mapped to neutral. All other emotions map directly.

**CREMA-D:** No surprise class — RAVDESS surprise examples are oversampled to compensate.

## Project Structure

```
moodmirror/
├── README.md
├── .gitignore
├── backend/
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py              # FastAPI endpoints
│   │   ├── schemas.py           # Pydantic models
│   │   ├── config.py            # All configuration
│   │   ├── inference/
│   │   │   ├── stt.py           # Speech-to-text (faster-whisper)
│   │   │   ├── text_emotion.py  # Text emotion classifier
│   │   │   ├── audio_emotion.py # Audio emotion classifier
│   │   │   ├── fusion.py        # Gated late fusion
│   │   │   └── response_generator.py
│   │   └── utils/
│   │       └── audio.py         # Audio conversion utilities
│   ├── models/                  # Trained weights (gitignored)
│   ├── data/                    # Datasets (gitignored)
│   └── training/
│       ├── download_datasets.py
│       ├── prepare_goemotions.py
│       ├── prepare_audio_datasets.py
│       ├── prepare_meld.py
│       ├── train_text_model.py
│       ├── train_audio_model.py
│       ├── train_fusion_model.py
│       └── evaluate.py
└── frontend/
    ├── package.json
    ├── index.html
    ├── vite.config.js
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── api.js
        ├── styles.css
        └── components/
            ├── Recorder.jsx
            ├── PredictionPanel.jsx
            ├── ProbabilityBars.jsx
            └── TranscriptBox.jsx
```

## API Reference

### `POST /api/analyze`

Upload an audio file for emotion analysis.

**Request:** `multipart/form-data` with field `file` (audio blob, max 10 MB, max 30 seconds)

**Response:**

```json
{
  "transcript": "I can't believe this happened again",
  "text_prediction": {
    "label": "anger",
    "confidence": 0.72,
    "probabilities": {
      "anger": 0.72, "disgust": 0.03, "fear": 0.05,
      "joy": 0.02, "neutral": 0.08, "sadness": 0.07, "surprise": 0.03
    }
  },
  "audio_prediction": {
    "label": "fear",
    "confidence": 0.54,
    "probabilities": { "...": "..." }
  },
  "fusion_prediction": {
    "label": "anger",
    "confidence": 0.79,
    "probabilities": { "...": "..." }
  },
  "spoken_response": "I heard frustration or anger. The words suggested anger, while the tone suggested fear."
}
```

### `GET /api/health`

Returns `{"status": "ok"}` when the server is running.

## Limitations and Ethics

- **Expressed emotion, not internal state.** The model estimates what emotion is *expressed* in speech, not what someone truly feels. These are different things.
- **STT errors propagate.** If Whisper mistranscribes a word, the text emotion prediction may be wrong. This is an inherent cascading-error risk in pipeline architectures.
- **Sarcasm is largely undetected.** Sarcastic speech often has contradictory text vs. tone signals. The model may split between them or get it wrong.
- **Domain bias.** Training data comes from Reddit comments (GoEmotions), acted speech recordings (RAVDESS, CREMA-D), and TV show dialogue (MELD). Performance on natural, spontaneous speech in other contexts will vary.
- **Confidence ≠ correctness.** A 90% confidence score means the model is 90% sure, not that it is 90% accurate. Neural network confidence is often poorly calibrated.
- **Cultural and linguistic bias.** All training data is in English. Emotion expression varies across cultures, languages, and individuals.

**Do not use this system for:**
- Mental health diagnosis or screening
- Hiring, disciplinary, or academic decisions
- Surveillance or behavioral monitoring
- Any context where the output affects someone's rights, opportunities, or wellbeing
