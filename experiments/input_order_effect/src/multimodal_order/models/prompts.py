from multimodal_order.constants import Condition


def qwen_messages(text: str, image_path: str, condition: Condition) -> list[dict]:
    image = {"type": "image", "image": image_path}
    words = {"type": "text", "text": text}
    content = [image, words] if condition == Condition.IMAGE_FIRST else [words, image]
    return [{"role": "user", "content": content}]


def internvl_prompt(text: str, condition: Condition) -> str:
    return f"<image>\n{text}" if condition == Condition.IMAGE_FIRST else f"{text}\n<image>"
