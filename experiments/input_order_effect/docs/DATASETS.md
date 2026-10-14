# Datasets

The repository includes fixed, category-stratified 500-example manifests selected with seed 42. The code expects users to obtain the corresponding images from the official benchmark sources.

## SafeBench (Typo)

Obtain the [official FigStep repository](https://github.com/ThuCCSLab/FigStep) and set `safe_bench_root` to its root. The loader expects:

```text
data/question/safebench.csv
data/images/SafeBench/query_ForbidQI_<category_id>_<task_id>_6.png
```

FigStep code is MIT licensed. Its data README contains a harmful-content warning and notes that some experimental files are not redistributed. Users must obtain the benchmark from its official source and follow its terms.

The harmful CSV question is retained for safety evaluation. Model inference uses the original benign FigStep list-completion prompt; the harmful instruction remains embedded in the typography image.

## MM-SafetyBench

Obtain MM-SafetyBench from its [official repository](https://github.com/isXinLiu/MM-SafetyBench) or [official dataset distribution](https://huggingface.co/datasets/PKU-Alignment/MM-SafetyBench) and set `mm_safety_bench_root`. Arrange selected images as `images/sd_typo/<filename>` and `images/sd/<filename>` beneath that root. The shipped manifests list the exact 500 selected examples.

Dataset images are deliberately excluded from this repository. Redistribution rights remain with the original dataset authors; users are responsible for complying with the dataset license and terms.

## Alpaca baseline

Set `alpaca_prompts` to a local JSON file containing `instruction` fields. The first 100 entries are used for the harmless reference distribution.
