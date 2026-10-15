# Reproducibility checklist

- Python dependencies are version-pinned in `requirements.txt`.
- Model revisions are pinned in `configs/experiment.json`.
- Dataset subsampling uses a fixed seed and category stratification.
- Output metadata records model, revision, seed, and decoding strategy.
- Manifests contain dataset-relative image paths.
- Responses and activations use only `image_first` and `text_first` condition names.
- Checkpoints, source images, and large activation arrays are excluded from Git.
- `scripts/validate_artifacts.py` checks counts, IDs, empty responses, and condition names.
