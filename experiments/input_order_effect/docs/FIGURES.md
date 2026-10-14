# Figures

`scripts/generate_figures.py` maps outputs to the paper numbering:

- Figure 2: `figure2_wildguard_asr.pdf` and `.png`
- Figure 3: `figure3_layer_cosine.pdf` and `.png`
- Figure 4: `figure4_pca.pdf` and `.png`

Figure 3 uses the mean Alpaca activation as the harmless/refusal reference and plots the Image First and Text First representation changes relative to the text-only input. Figure 4 fits PCA on harmless and harmful text-only baselines, then projects both input-order conditions.

Activation arrays are not tracked in Git. Store or distribute them as a versioned release artifact with checksums if exact figure reproduction without model inference is required.
