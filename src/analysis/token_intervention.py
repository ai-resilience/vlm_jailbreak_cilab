#!/usr/bin/env python3
"""Token intervention: Mask attention from user_input_end_pos tokens to image tokens.

Goal: Set attention from tokens at positions >= user_input_end_pos to image_token_indices to -inf.
Debug: Compare baseline attention (before masking) with masked attention (after masking).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
from collections import defaultdict
from src.models.base import find_layers


class TokenInterventionHook:
    """Hook to mask attention from user_input_end_pos tokens to image tokens."""
    
    def __init__(
        self,
        user_input_end_pos: Optional[int] = None,
        image_token_indices: Optional[List[int]] = None,
        mask_only_decoding: bool = False
    ):
        """
        Args:
            user_input_end_pos: Token position where user input ends
            image_token_indices: List of image token indices to mask
            mask_only_decoding: If True, only mask during decoding stage (not prefilling)
        """
        self.user_input_end_pos = user_input_end_pos
        self.image_token_indices = image_token_indices or []
        self.mask_only_decoding = mask_only_decoding
        
        # Storage for attention weights (for debugging)
        self.attention_weights = defaultdict(list)  # {layer_idx: [attn_weights]}
        self._baseline_captured = False  # Track if samples have been printed
        
        # Hooks
        self.hooks = []
        self.attention_wrappers = {}  # {layer_idx: (attn_module, original_forward)}
    
    def _wrap_attention_forward(self, layer_idx: int, attn_module: nn.Module):
        """Wrap attention module's forward to mask attention scores.
        
        Masks attention from query positions >= user_input_end_pos to image_token_indices.
        Only works during prefilling (when past_key_values is None).
        """
        original_forward = attn_module.forward
        
        def masked_attention_forward(*args, **kwargs):
            """Wrapped attention forward that masks image tokens."""
            # Early return if no masking needed
            if self.user_input_end_pos is None or not self.image_token_indices:
                return original_forward(*args, **kwargs)
            
            # Get hidden_states to determine seq_len_q
            hidden_states = None
            if len(args) > 0 and isinstance(args[0], torch.Tensor):
                hidden_states = args[0]
            elif 'hidden_states' in kwargs:
                hidden_states = kwargs['hidden_states']
            
            if hidden_states is None or len(hidden_states.shape) < 2:
                return original_forward(*args, **kwargs)
            
            seq_len_q = hidden_states.shape[1]
            
            # Get attention_mask from kwargs (LlamaDecoderLayer calls with keyword arguments)
            attention_mask = kwargs.get('attention_mask', None)
            
            # Check if decoding step
            # 1. past_key_value가 있으면 디코딩 (일반적인 경우)
            # 2. attention_mask shape이 [batch, 1, 1, seq_len_k]이고 seq_len_q == 1이면 디코딩
            #    (디코딩 첫 스텝에서 past_key_value가 None일 수 있음)
            past_key_value = kwargs.get('past_key_value', None)
            is_decoding_by_cache = past_key_value is not None
            
            # attention_mask shape으로도 확인: [batch, 1, 1, seq_len_k] 형태이고 seq_len_q == 1이면 디코딩
            is_decoding_by_mask = False
            if attention_mask is not None and len(attention_mask.shape) == 4:
                # [batch, 1, 1, seq_len_k] 형태이고 seq_len_q == 1이면 디코딩
                if attention_mask.shape[1] == 1 and attention_mask.shape[2] == 1 and seq_len_q == 1:
                    # seq_len_k가 seq_len_q보다 크면 디코딩 (prefilling에서는 보통 같음)
                    seq_len_k_from_mask = attention_mask.shape[-1]
                    if seq_len_k_from_mask > seq_len_q:
                        is_decoding_by_mask = True
            
            is_decoding = is_decoding_by_cache or is_decoding_by_mask
            
            # Prefilling 단계: attention_mask를 직접 수정 (mask_only_decoding이 False일 때만)

            if attention_mask.shape[2] == 1:
                a = None

            if not is_decoding and not self.mask_only_decoding and attention_mask is not None:
                attention_mask = attention_mask.clone()
                
                if len(attention_mask.shape) == 4:
                    seq_len_k = attention_mask.shape[-1]
                    
                    # Handle [batch, 1, 1, seq_len_k] -> expand to [batch, 1, seq_len_q, seq_len_k]
                    if attention_mask.shape[1] == 1 and attention_mask.shape[2] == 1:
                        mask_expanded = attention_mask.repeat(1, 1, seq_len_q, 1)
                        
                        # Apply masking: set attention from query_pos >= user_input_end_pos to image_token_indices to -inf
                        for q_pos in range(seq_len_q):
                            if q_pos >= self.user_input_end_pos:
                                for img_idx in self.image_token_indices:
                                    if 0 <= img_idx < seq_len_k:
                                        mask_expanded[:, :, q_pos, img_idx] = float('-inf')
                        
                        kwargs['attention_mask'] = mask_expanded
                    
                    # Handle [batch, 1, seq_len_q, seq_len_k] - already in correct shape
                    elif attention_mask.shape[1] == 1 and attention_mask.shape[2] == seq_len_q:
                        # Apply masking
                        for q_pos in range(seq_len_q):
                            if q_pos >= self.user_input_end_pos:
                                for img_idx in self.image_token_indices:
                                    if 0 <= img_idx < seq_len_k:
                                        attention_mask[:, :, q_pos, img_idx] = float('-inf')
                        
                        kwargs['attention_mask'] = attention_mask
                
                kwargs['output_attentions'] = True
                return original_forward(*args, **kwargs)
            
            # Prefilling 단계이지만 mask_only_decoding이 True인 경우: 마스킹 없이 진행 (output_attentions도 False로 설정하여 속도 향상)
            elif not is_decoding and self.mask_only_decoding:
                # output_attentions를 설정하지 않아서 속도 향상
                return original_forward(*args, **kwargs)
            
            # Decoding 단계: attention_mask를 동적으로 생성하여 수정
            elif is_decoding:
                device = hidden_states.device
                dtype = hidden_states.dtype
                batch_size = hidden_states.shape[0]
                
                # past_key_value에서 전체 sequence length 가져오기
                cache_position = kwargs.get('cache_position', None)
                
                # key_states의 실제 길이를 추정 (past_key_value에서)
                try:
                    if hasattr(past_key_value, 'get_seq_length'):
                        seq_len_k = past_key_value.get_seq_length()
                    elif hasattr(past_key_value, 'key_cache'):
                        # DynamicCache의 경우
                        if len(past_key_value.key_cache) > 0:
                            # 첫 번째 레이어의 key cache shape 확인
                            cached_key = past_key_value.key_cache[0]
                            if isinstance(cached_key, tuple):
                                cached_key = cached_key[0]
                            seq_len_k = cached_key.shape[-2] if len(cached_key.shape) >= 2 else cached_key.shape[-1]
                        else:
                            seq_len_k = self.user_input_end_pos + len(self.image_token_indices) + 1
                    elif cache_position is not None:
                        seq_len_k = cache_position.max().item() + 1
                    else:
                        # 기본값: prefilling 길이 + 현재까지 생성된 토큰 수 (부정확할 수 있음)
                        seq_len_k = self.user_input_end_pos + len(self.image_token_indices) + 1
                except Exception as e:
                    # Fallback: cache_position 사용
                    if cache_position is not None:
                        seq_len_k = cache_position.max().item() + 1
                    else:
                        seq_len_k = self.user_input_end_pos + len(self.image_token_indices) + 1
                
                # 기존 attention_mask가 있으면 그것을 기반으로 수정, 없으면 새로 생성
                # 디코딩 단계에서는 모델이 이미 causal mask를 포함한 attention_mask를 생성했을 수 있음
                if attention_mask is not None:
                    # 기존 mask를 clone하여 수정
                    attention_mask = attention_mask.clone()
                    
                    # 디코딩 단계에서 attention_mask shape 확인
                    # attention_mask가 [batch, 1, 1, seq_len_k] 형태면 seq_len_q = 1이므로 q_pos = 0만 가능
                    if len(attention_mask.shape) == 4:
                        # attention_mask의 실제 query dimension 확인
                        actual_seq_len_q = attention_mask.shape[2]
                        
                        # 디코딩 단계에서는 보통 seq_len_q = 1이므로 q_pos = 0
                        if actual_seq_len_q == 1:
                            # seq_len_q = 1이면 항상 q_pos = 0
                            q_pos = 0
                            for img_idx in self.image_token_indices:
                                if 0 <= img_idx < seq_len_k and img_idx < attention_mask.shape[-1]:
                                    attention_mask[:, :, q_pos, img_idx] = float('-inf')
                        else:
                            # seq_len_q > 1인 경우 (드물지만 가능)
                            # 마지막 query position (새 토큰)에서 마스킹
                            for img_idx in self.image_token_indices:
                                if 0 <= img_idx < seq_len_k and img_idx < attention_mask.shape[-1]:
                                    attention_mask[:, :, -1, img_idx] = float('-inf')
                    else:
                        # attention_mask shape이 4D가 아니면 그대로 진행
                        pass
                else:
                    # attention_mask가 None이면 새로 생성 (causal mask는 모델이 자동 처리)
                    # 디코딩 단계에서는 seq_len_q = 1 (새 토큰)
                    attention_mask = torch.zeros(
                        (batch_size, 1, seq_len_q, seq_len_k),
                        device=device,
                        dtype=dtype
                    )
                    
                    # 새 토큰(seq_len_q=1, 즉 q_pos=0)에서 이미지 토큰으로의 attention 마스킹
                    for q_pos in range(seq_len_q):
                        for img_idx in self.image_token_indices:
                            if 0 <= img_idx < seq_len_k:
                                attention_mask[:, :, q_pos, img_idx] = float('-inf')
                
                kwargs['attention_mask'] = attention_mask
                kwargs['output_attentions'] = True
                return original_forward(*args, **kwargs)
            
            # attention_mask가 None이고 prefilling 단계가 아니면 그대로 진행
            return original_forward(*args, **kwargs)
        
        attn_module.forward = masked_attention_forward
        self.attention_wrappers[layer_idx] = (attn_module, original_forward)
    
    def _print_attention_samples(self, layer_idx: int, attn_weights: torch.Tensor):
        """Print sample attention values for masked attention (디코딩 단계에서만 출력)."""
        if len(attn_weights.shape) < 3:
            return
        
        # Get shapes
        if len(attn_weights.shape) == 4:
            batch, num_heads, seq_len_q, seq_len_k = attn_weights.shape
            use_heads = True
        else:
            batch, seq_len_q, seq_len_k = attn_weights.shape
            use_heads = False
        
        # mask_only_decoding이 True일 때는 디코딩 단계(seq_len_q == 1)에서만 출력
        if self.mask_only_decoding:
            if seq_len_q != 1:  # 디코딩 단계가 아님 (prefilling)
                return
            # 디코딩 단계: q_pos = 0 (새로 생성되는 토큰)
            q_pos = 0
        else:
            # mask_only_decoding이 False일 때는 prefilling 단계에서도 출력
            if self.user_input_end_pos is None or self.user_input_end_pos >= seq_len_q:
                return
            q_pos = self.user_input_end_pos
        
        if q_pos >= seq_len_q:
            return
        
        # Get image token range
        img_start = self.image_token_indices[0] if self.image_token_indices else 0
        img_end = min(self.image_token_indices[-1] + 1, seq_len_k) if self.image_token_indices else seq_len_k
        
        # Extract attention to image tokens
        if use_heads:
            # Use first head
            head_idx = 0
            masked_img_attn = attn_weights[0, head_idx, q_pos, img_start:img_end]
        else:
            masked_img_attn = attn_weights[0, q_pos, img_start:img_end]
        
        # 샘플 토큰 선택 (처음 5개 이미지 토큰)
        num_samples = min(5, len(masked_img_attn))
        sample_indices = list(range(num_samples))
        
        # 한번만 출력 (첫 번째 레이어에서만)
        if layer_idx == 0 and not self._baseline_captured:
            stage = "Decoding" if self.mask_only_decoding else "Prefilling"
            print(f"\n{'='*60}")
            print(f"📊 Masked Attention Samples (Layer {layer_idx}, {stage} Stage, Query Position {q_pos})")
            print(f"{'='*60}")
            print(f"Image tokens: {img_start} to {img_end-1}")
            print(f"Sample image token indices: {[img_start + i for i in sample_indices]}")
            print(f"\nMasked attention values:")
            for i in sample_indices:
                val = masked_img_attn[i].item()
                if torch.isinf(torch.tensor(val)) or val < -1e6:
                    print(f"  Token {img_start + i}: -inf (masked)")
                else:
                    print(f"  Token {img_start + i}: {val:.6f}")
            print(f"{'='*60}\n")
            self._baseline_captured = True
    
    def _get_attention_hook(self, layer_idx: int):
        """Hook to extract and print attention weights."""
        def attention_hook(module, input, output):
            """Extract attention weights and print samples."""
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]  # [batch, num_heads, seq_len_q, seq_len_k] or [batch, seq_len_q, seq_len_k]
                if attn_weights is not None:
                    self.attention_weights[layer_idx].append(attn_weights.detach().clone())
                    
                    # Print samples (한번만 출력)
                    if layer_idx == 0:
                        self._print_attention_samples(layer_idx, attn_weights)
            
            return output
        
        return attention_hook
    
    def register_hooks(self, model: nn.Module, model_name: str):
        """Register hooks on attention layers."""
        self.hooks = []
        
        # Find transformer layers
        try:
            layers = find_layers(model)
        except ValueError as e:
            print(f"⚠️  {e}")
            return
        
        # Register hooks on each layer
        for layer_idx, layer in enumerate(layers):
            # Find attention module
            if hasattr(layer, 'self_attn'):
                attn_module = layer.self_attn
            elif hasattr(layer, 'attention'):
                attn_module = layer.attention
            else:
                continue
            
            # Register attention hook (to extract attention weights)
            hook_handle = attn_module.register_forward_hook(
                self._get_attention_hook(layer_idx)
            )
            self.hooks.append(hook_handle)
            
            # Wrap attention forward to mask image tokens
            if self.user_input_end_pos is not None and self.image_token_indices:
                self._wrap_attention_forward(layer_idx, attn_module)
    
    def remove_hooks(self):
        """Remove all registered hooks and restore original forward methods."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Restore original attention forward methods
        for layer_idx, (attn_module, original_forward) in self.attention_wrappers.items():
            attn_module.forward = original_forward
        self.attention_wrappers.clear()
    
    def get_attention_weights(self, layer_idx: Optional[int] = None) -> Dict:
        """Get extracted attention weights."""
        if layer_idx is not None:
            return {layer_idx: self.attention_weights[layer_idx]}
        return dict(self.attention_weights)
    
    def print_comparison_summary(self, max_layers: int = 5):
        """Print detailed comparison summary between baseline and masked attention."""
        if not self.comparison_stats:
            print("⚠️  No comparison statistics available.")
            return
        
        print(f"\n{'='*80}")
        print(f"📊 ATTENTION MASKING COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"User input ends at position: {self.user_input_end_pos}")
        print(f"Image tokens: {self.image_token_indices[0]} to {self.image_token_indices[-1]} (total: {len(self.image_token_indices)})")
        print(f"{'='*80}\n")
        
        for layer_idx in sorted(self.comparison_stats.keys())[:max_layers]:
            stats = self.comparison_stats[layer_idx]
            
            print(f"\n{'─'*80}")
            print(f"Layer {layer_idx}")
            print(f"{'─'*80}")
            
            for pos_stats in stats['sample_positions']:
                q_pos = pos_stats['q_pos']
                baseline = pos_stats['baseline']
                masked = pos_stats['masked']
                reduction = pos_stats['reduction']
                
                print(f"\n  Query Position {q_pos}:")
                print(f"    BEFORE masking:")
                print(f"      Mean: {baseline['mean']:.6f} | Max: {baseline['max']:.6f} | Min: {baseline['min']:.6f} | Std: {baseline['std']:.6f}")
                print(f"    AFTER masking:")
                print(f"      Mean: {masked['mean']:.6f} | Max: {masked['max']:.6f} | Min: {masked['min']:.6f} | Std: {masked['std']:.6f}")
                print(f"    Reduction: {reduction['ratio']:.2f}x ({reduction['percentage']:.1f}% decrease)")
        
        print(f"\n{'='*80}\n")
    
    def get_comparison_stats(self, layer_idx: Optional[int] = None) -> Dict:
        """Get comparison statistics."""
        if layer_idx is not None:
            return {layer_idx: self.comparison_stats.get(layer_idx)}
        return dict(self.comparison_stats)


def find_image_token_range(
    model_name: str,
    processor: Any,
    input_ids: torch.Tensor,
    images_seq_mask: Optional[torch.Tensor] = None
) -> Optional[tuple]:
    """Find the start and end indices of image tokens in the input sequence."""
    # Handle batch dimension
    if len(input_ids.shape) > 1:
        input_ids = input_ids[0]
    
    # Method 1: Use images_seq_mask if available
    if images_seq_mask is not None:
        if len(images_seq_mask.shape) > 1:
            images_seq_mask = images_seq_mask[0]
        
        image_indices = torch.where(images_seq_mask)[0]
        if len(image_indices) > 0:
            return (int(image_indices[0].item()), int(image_indices[-1].item()) + 1)
    
    # Method 2: Find image token IDs in the sequence
    image_token_ids = []
    image_token_candidates = [
        "<image>", "<img>", "</img>", "<image_1>", "<image_start>",
        "<image_patch>", "<IMG_CONTEXT>", "<|image_pad|>"
    ]
    
    for token_str in image_token_candidates:
        try:
            token_id = processor.tokenizer.convert_tokens_to_ids(token_str)
            if token_id is not None and token_id != processor.tokenizer.unk_token_id:
                image_token_ids.append(token_id)
        except:
            continue
    
    # Also try to get image token ID from processor
    if hasattr(processor, 'image_token_id'):
        image_token_ids.append(processor.image_token_id)
    if hasattr(processor, 'image_id'):
        image_token_ids.append(processor.image_id)
    
    # Find image token positions
    image_positions = []
    for token_id in image_token_ids:
        positions = torch.where(input_ids == token_id)[0]
        image_positions.extend(positions.tolist())
    
    if image_positions:
        image_positions = sorted(set(image_positions))
        if len(image_positions) > 0:
            start = image_positions[0]
            end = image_positions[-1] + 1
            expected_range = set(range(start, end))
            if set(image_positions).issubset(expected_range):
                return (start, end)
            else:
                return (min(image_positions), max(image_positions) + 1)
    
    return None


def find_user_input_end_pos(
    model_name: str,
    processor: Any,
    input_ids: torch.Tensor,
    prompt: str
) -> Optional[int]:
    """Find the token position where user input ends (for LLaVA models)."""
    if len(input_ids.shape) > 1:
        input_ids = input_ids[0]
    
    decoded_text = processor.tokenizer.decode(input_ids, skip_special_tokens=False)
    
    # Find ASSISTANT: token position
    assistant_token_candidates = ["ASSISTANT:", "ASSISTANT :", "assistant:", "assistant :"]
    for token_str in assistant_token_candidates:
        try:
            token_ids = processor.tokenizer.encode(token_str, add_special_tokens=False)
            if len(token_ids) > 0:
                token_ids_tensor = torch.tensor(token_ids, device=input_ids.device, dtype=input_ids.dtype)
                for i in range(len(input_ids) - len(token_ids) + 1):
                    if torch.equal(input_ids[i:i+len(token_ids)], token_ids_tensor):
                        # User input ends just before ASSISTANT:
                        return i - 1
        except:
            continue
    
    # Fallback: try to find prompt text end position
    if prompt and prompt.strip() in decoded_text:
        prompt_end_char = decoded_text.find(prompt.strip()) + len(prompt.strip())
        text_before_end = decoded_text[:prompt_end_char]
        tokens_before = processor.tokenizer.encode(text_before_end, add_special_tokens=False)
        return len(tokens_before) - 1
    
    return None


def generate_with_token_intervention(
    model: Any,
    processor: Any,
    model_name: str,
    prompt: str,
    img: Optional[str] = None,
    mask_image_tokens: bool = False,
    mask_only_decoding: bool = False,
    max_new_tokens: int = 128,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Generate response with token intervention.
    
    Args:
        model: The VLM model
        processor: Model processor
        model_name: Name of the model
        prompt: Text prompt
        img: Path to image (optional)
        mask_image_tokens: Whether to mask all image tokens
        mask_only_decoding: If True, only mask during decoding stage (not prefilling)
        max_new_tokens: Maximum number of tokens to generate
        output_dir: Directory to save outputs (optional)
        
    Returns:
        Dictionary containing:
            - response: Generated text
            - attention_weights: Attention weights (for debugging)
    """
    from src.inference.processor import build_prompt
    
    print(f"\n{'='*60}")
    print(f"Stage 1: Preparing inputs")
    print(f"{'='*60}")
    
    # Prepare inputs
    inputs, attention_mask = build_prompt(model, processor, model_name, img, prompt)
    
    # Get input_ids
    if isinstance(inputs, dict) and "input_ids" in inputs:
        input_ids = inputs["input_ids"]
    elif hasattr(inputs, 'input_ids'):
        input_ids = inputs.input_ids
    else:
        input_ids = None
    
    # Get images_seq_mask if available
    images_seq_mask = None
    if hasattr(attention_mask, 'images_seq_mask'):
        images_seq_mask = attention_mask.images_seq_mask
    elif isinstance(attention_mask, dict) and 'images_seq_mask' in attention_mask:
        images_seq_mask = attention_mask['images_seq_mask']
    
    print(f"Stage 2: Finding user_input_end_pos and image_token_indices")
    print(f"{'='*60}")
    
    # Find user input end position and image token indices
    user_input_end_pos = None
    image_token_indices = []
    
    if mask_image_tokens:
        # Find all image tokens
        image_range = find_image_token_range(model_name, processor, input_ids, images_seq_mask)
        if image_range:
            start_idx, end_idx = image_range
            image_token_indices = list(range(start_idx, end_idx))
            print(f"✅ Found image tokens: indices {start_idx} to {end_idx-1} (total: {len(image_token_indices)} tokens)")
        else:
            print("⚠️  Could not find image token range.")
        
        # Find user input end position
        if model_name in ["llava"]:
            user_input_end_pos = find_user_input_end_pos(model_name, processor, input_ids, prompt)
            if user_input_end_pos is not None:
                print(f"✅ User input ends at token position {user_input_end_pos}")
            else:
                print("⚠️  Could not find user input end position.")
    
    if user_input_end_pos is None or not image_token_indices:
        print("⚠️  Cannot apply masking: missing user_input_end_pos or image_token_indices")
        mask_image_tokens = False
    
    if mask_image_tokens:
        if mask_only_decoding:
            print(f"📌 Masking mode: Decoding stage only (prefilling will not be masked)")
        else:
            print(f"📌 Masking mode: Both prefilling and decoding stages")
    
    print(f"\n{'='*60}")
    print(f"Stage 3: Registering hooks and generating")
    print(f"{'='*60}")
    
    # Create intervention hook
    intervention = TokenInterventionHook(
        user_input_end_pos=user_input_end_pos,
        image_token_indices=image_token_indices,
        mask_only_decoding=mask_only_decoding
    )
    
    # Register hooks
    intervention.register_hooks(model, model_name)
    
    try:
        # Generate
        if model_name in ["deepseek", "deepseek2"]:
            inputs.to(model.device, dtype=torch.bfloat16)
            attention_mask = attention_mask.to(model.device)
            
            language_model = model.language_model if model_name == "deepseek" else model.language
            
            with torch.no_grad():
                out = language_model.generate(
                    inputs_embeds=inputs,
                    attention_mask=attention_mask,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    bos_token_id=processor.tokenizer.bos_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True
                )
                generated_ids = out.sequences[0] if hasattr(out, 'sequences') else out[0]
        
        else:  # llava, llava_next, qwen, intern, kimi
            inputs.to(model.device, dtype=torch.bfloat16)
            input_len = inputs["input_ids"].shape[-1]
            
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True
                )
                generated_ids = out.sequences[0][input_len:] if hasattr(out, 'sequences') else out[0][input_len:]
        
        # Decode response
        response = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Get attention weights (for debugging)
        attention_weights = intervention.get_attention_weights()
        
        return {
            'response': response,
            'attention_weights': attention_weights
        }
    
    finally:
        # Always remove hooks
        intervention.remove_hooks()
