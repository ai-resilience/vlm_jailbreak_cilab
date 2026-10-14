import numpy as np
import torch

from multimodal_order.activations import last_nonpadding_indices, refusal_direction
from multimodal_order.datasets import stratified_sample
from multimodal_order.datasets.manifests import FIGSTEP_PROMPT
from multimodal_order.evaluation.metrics import is_target_string, majority_vote
from multimodal_order.models.prompts import qwen_messages
from multimodal_order.constants import Condition
from multimodal_order.reproducibility import greedy_generation_kwargs
from multimodal_order.text import decode_without_thinking, strip_thinking


def test_last_token_left_and_right_padding():
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 1, 0]])
    assert last_nonpadding_indices(mask).tolist() == [3, 1, 2]


def test_named_input_order():
    first = qwen_messages("text", "image.png", Condition.IMAGE_FIRST)[0]["content"]
    last = qwen_messages("text", "image.png", Condition.TEXT_FIRST)[0]["content"]
    assert [x["type"] for x in first] == ["image", "text"]
    assert [x["type"] for x in last] == ["text", "image"]
    assert "list numbered 1, 2, and 3" in FIGSTEP_PROMPT


def test_thinking_is_excluded_and_target_is_shared():
    assert strip_thinking("<think>hidden</think>1. actionable") == "1. actionable"
    assert is_target_string("<think>reasoning</think>\nStep 1: do this")


def test_decode_strips_thinking_before_special_tokens():
    class Tokenizer:
        all_special_tokens = ["<think>", "</think>", "<|im_end|>"]

        @staticmethod
        def decode(_ids, skip_special_tokens=False):
            assert skip_special_tokens is False
            return "<think>private reasoning</think>Final answer<|im_end|>"

    assert decode_without_thinking(Tokenizer(), [1, 2]) == "Final answer"


def test_exact_majority_vote():
    assert majority_vote(True, False, True)
    assert not majority_vote(True, False, False)


def test_stratified_500_and_greedy():
    rows = [{"category_name": str(i % 13), "category_id": i % 13, "task_id": i} for i in range(520)]
    selected = stratified_sample(rows, 500, 42)
    assert len(selected) == 500 and selected == stratified_sample(rows, 500, 42)
    assert greedy_generation_kwargs(32) == {"do_sample": False, "max_new_tokens": 32}


def test_refusal_direction_shape():
    assert refusal_direction(np.ones((100, 3, 4))).shape == (3, 4)
