from enum import Enum


class Condition(str, Enum):
    IMAGE_FIRST = "image_first"
    TEXT_FIRST = "text_first"


CONDITIONS = (Condition.IMAGE_FIRST, Condition.TEXT_FIRST)
CONDITION_LABELS = {Condition.IMAGE_FIRST: "Image First", Condition.TEXT_FIRST: "Text First"}
BENCHMARKS = ("safebench_typo", "mmsafetybench_sd_typo", "mmsafetybench_sd")
