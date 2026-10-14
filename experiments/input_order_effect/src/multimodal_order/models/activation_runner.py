from pathlib import Path

import numpy as np
from PIL import Image

from multimodal_order.activations import extract_last_input_token
from multimodal_order.constants import CONDITIONS, Condition
import multimodal_order.models.internvl as internvl
from multimodal_order.models.prompts import qwen_messages
from multimodal_order.models.runner import InferenceRunner


class ActivationRunner(InferenceRunner):
    def _text_activation(self, text: str) -> np.ndarray:
        import torch
        if self.spec["family"] == "internvl":
            prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                output = self.model.language_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, output_hidden_states=True, return_dict=True, use_cache=False)
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[prompt], padding=True, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                output = self.model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        return extract_last_input_token(output.hidden_states, inputs.attention_mask)[0]

    def _qwen_activation(self, row: dict, condition: Condition) -> np.ndarray:
        from qwen_vl_utils import process_vision_info
        import torch
        messages = qwen_messages(row["instruction"], row["image_path"], condition)
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = self.processor(text=[rendered], images=images, videos=videos, padding=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        return extract_last_input_token(output.hidden_states, inputs.attention_mask)[0]

    def _internvl_activation(self, row: dict, condition: Condition) -> np.ndarray:
        from torchvision.transforms import Compose, InterpolationMode, Lambda, Normalize, Resize, ToTensor
        import torch
        transform = Compose([Lambda(lambda x: x.convert("RGB")), Resize((448, 448), interpolation=InterpolationMode.BICUBIC), ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        pixels = transform(Image.open(row["image_path"])).unsqueeze(0).to(dtype=torch.bfloat16, device=self.model.device)
        prompt = internvl.serialize_prompt(self.model, row["instruction"], condition == Condition.IMAGE_FIRST)
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = internvl.forward(self.model, encoded.input_ids, pixels, encoded.attention_mask)
        return extract_last_input_token(output.hidden_states, encoded.attention_mask)[0]

    def run_activations(self, rows: list[dict], alpaca_prompts: list[str], output_dir: Path) -> None:
        if self.model is None:
            self.load()
        output_dir.mkdir(parents=True, exist_ok=True)
        for condition in CONDITIONS:
            values = []
            for row in rows:
                fn = self._internvl_activation if self.spec["family"] == "internvl" else self._qwen_activation
                values.append(fn(row, condition))
            np.save(output_dir / f"{condition.value}.npy", np.stack(values).astype(np.float32))
        text_only = [self._text_activation(row.get("question", row["instruction"])) for row in rows]
        harmless = [self._text_activation(text) for text in alpaca_prompts[:100]]
        np.save(output_dir / "text_only.npy", np.stack(text_only).astype(np.float32))
        np.save(output_dir / "alpaca.npy", np.stack(harmless).astype(np.float32))
