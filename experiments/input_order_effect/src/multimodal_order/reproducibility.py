import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        from transformers import set_seed as transformers_set_seed

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        transformers_set_seed(seed)
    except ImportError:
        pass


def greedy_generation_kwargs(max_new_tokens: int) -> dict:
    """Greedy decoding; temperature is intentionally not passed to generate()."""
    return {"do_sample": False, "max_new_tokens": int(max_new_tokens)}
