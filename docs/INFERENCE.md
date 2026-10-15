# Inference

`scripts/run_inference.py` evaluates both named conditions:

- **Image First**: image content precedes text content.
- **Text First**: text content precedes image content.

Outputs are stored as `image_first.json` and `text_first.json`. Each file records the dataset, model identifier, pinned model revision, seed, and decoding metadata. Qwen models use ordered multimodal content lists; InternVL uses an explicitly serialized image-token block in the corresponding position.

The scripts do not download or bundle checkpoints. Update model IDs only when intentionally reproducing a different revision.
