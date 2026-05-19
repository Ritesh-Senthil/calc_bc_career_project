# Models Directory

Trained model weights are stored here. These directories are not checked into version control.

## Expected Structure

- `text_emotion/` — Fine-tuned DeBERTa-v3-base text emotion classifier
- `audio_emotion/` — Fine-tuned Wav2Vec2-base audio emotion classifier
- `fusion/` — Gated late-fusion classifier

If any directory is missing, the application falls back to public pretrained models from Hugging Face.
