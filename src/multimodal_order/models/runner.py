import json
from pathlib import Path

from PIL import Image

from multimodal_order.constants import CONDITIONS, CONDITION_LABELS, Condition
from multimodal_order.models.prompts import qwen_messages
import multimodal_order.models.internvl as internvl
from multimodal_order.reproducibility import greedy_generation_kwargs, set_seed
from multimodal_order.text import decode_without_thinking, strip_thinking


class InferenceRunner:
    def __init__(self, model_spec: dict, seed: int, max_new_tokens: int):
        self.spec = model_spec
        self.seed = seed
        self.generation = greedy_generation_kwargs(max_new_tokens)
        self.model = self.processor = self.tokenizer = None

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        set_seed(self.seed)
        common = {"revision": self.spec["revision"], "dtype": torch.bfloat16, "device_map": "auto"}
        family = self.spec["family"]
        if family == "internvl":
            self.model = AutoModel.from_pretrained(self.spec["id"], trust_remote_code=True, **common).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.spec["id"], revision=self.spec["revision"], trust_remote_code=True, use_fast=False)
            internvl.prepare(self.model, self.tokenizer)
        else:
            if family == "qwen25":
                from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
            else:
                from transformers import Qwen3VLForConditionalGeneration as ModelClass
            self.model = ModelClass.from_pretrained(self.spec["id"], **common).eval()
            # Pin the legacy image processor path so a Transformers default change
            # cannot silently alter the paper's preprocessing.
            self.processor = AutoProcessor.from_pretrained(
                self.spec["id"], revision=self.spec["revision"], use_fast=False
            )

        # Some checkpoints store sampling-only values even though generation is
        # greedy. Clear them to prevent misleading validation warnings.
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None

    def _qwen(self, row: dict, condition: Condition) -> tuple[str, str]:
        from qwen_vl_utils import process_vision_info

        messages = qwen_messages(row["instruction"], row["image_path"], condition)
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = self.processor(text=[rendered], images=images, videos=videos, padding=True, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, **self.generation)
        generated = output[:, inputs.input_ids.shape[1]:]
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        return rendered, decode_without_thinking(tokenizer, generated[0])

    def _internvl(self, row: dict, condition: Condition) -> tuple[str, str]:
        from torchvision.transforms import Compose, InterpolationMode, Lambda, Normalize, Resize, ToTensor
        import torch

        transform = Compose([Lambda(lambda x: x.convert("RGB")), Resize((448, 448), interpolation=InterpolationMode.BICUBIC), ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        pixels = transform(Image.open(row["image_path"])).unsqueeze(0).to(dtype=torch.bfloat16, device=self.model.device)
        prompt = internvl.serialize_prompt(self.model, row["instruction"], condition == Condition.IMAGE_FIRST)
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        eos_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        generated = internvl.generate(self.model, encoded.input_ids, pixels, attention_mask=encoded.attention_mask, eos_token_id=eos_id, pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id, **self.generation)
        # inputs_embeds generation may return only new tokens; decode its returned sequence directly.
        answer = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return prompt, strip_thinking(answer)

    def run(self, rows: list[dict], output_dir: Path, model_name: str, benchmark: str) -> None:
        if self.model is None:
            self.load()
        output_dir.mkdir(parents=True, exist_ok=True)
        for condition in CONDITIONS:
            records = []
            for row in rows:
                prompt, response = self._internvl(row, condition) if self.spec["family"] == "internvl" else self._qwen(row, condition)
                records.append({"category_id": row.get("category_id"), "task_id": row.get("task_id"), "question": row.get("question", ""), "instruction": row["instruction"], "condition": CONDITION_LABELS[condition], "response": response})
            payload = {"metadata": {"benchmark": benchmark, "model": model_name, "model_id": self.spec["id"], "revision": self.spec["revision"], "seed": self.seed, "decoding_strategy": "greedy", "temperature": 0.0}, "rows": records}
            (output_dir / f"{condition.value}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
