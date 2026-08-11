def prepare(model, tokenizer) -> None:
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")


def image_block(model) -> str:
    return "<img>" + "<IMG_CONTEXT>" * int(getattr(model, "num_image_token", 256)) + "</img>"


def serialize_prompt(model, text: str, image_first: bool) -> str:
    block = image_block(model)
    content = f"{block}\n{text}" if image_first else f"{text}\n{block}"
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


def merge_image_embeddings(model, input_ids, pixel_values):
    vision = model.extract_feature(pixel_values)
    embeddings = model.language_model.get_input_embeddings()(input_ids)
    batch, sequence, hidden = embeddings.shape
    flat = embeddings.reshape(batch * sequence, hidden)
    selected = input_ids.reshape(-1) == model.img_context_token_id
    if int(selected.sum()) != vision.numel() // hidden:
        raise ValueError("image-token count does not match InternVL vision embeddings")
    flat[selected] = vision.reshape(-1, hidden).to(device=flat.device, dtype=flat.dtype)
    return flat.reshape(batch, sequence, hidden)


def forward(model, input_ids, pixel_values, attention_mask=None):
    embeddings = merge_image_embeddings(model, input_ids, pixel_values)
    return model.language_model(inputs_embeds=embeddings, attention_mask=attention_mask, output_hidden_states=True, return_dict=True, use_cache=False)


def generate(model, input_ids, pixel_values, attention_mask=None, **kwargs):
    embeddings = merge_image_embeddings(model, input_ids, pixel_values)
    return model.language_model.generate(inputs_embeds=embeddings, attention_mask=attention_mask, **kwargs)
