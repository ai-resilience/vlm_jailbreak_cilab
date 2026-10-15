import re


_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_OPEN_THINK = re.compile(r"<think>.*$", flags=re.IGNORECASE | re.DOTALL)


def strip_thinking(text: str) -> str:
    value = _THINK_BLOCK.sub("", str(text or ""))
    value = _OPEN_THINK.sub("", value)
    return value.strip()


def decode_without_thinking(tokenizer, token_ids) -> str:
    """Remove reasoning before removing tokenizer special-token markers."""
    raw = tokenizer.decode(token_ids, skip_special_tokens=False)
    value = strip_thinking(raw)
    for token in sorted(set(tokenizer.all_special_tokens), key=len, reverse=True):
        value = value.replace(token, "")
    return value.strip()
