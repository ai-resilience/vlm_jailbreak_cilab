# flash-attn wheel이 torch와 ABI가 안 맞으면 import 단계에서 터진다. eager만 써도 transformers가
# flash_attn 존재 여부를 검사하며 깨진 .so를 로드할 수 있어, 메타데이터만 막아 sdpa/eager로 가게 함.
# 근본 해결: `pip uninstall flash-attn` 후 재실행, 또는 현재 torch에 맞게 flash-attn 재설치.
import importlib.metadata

_orig_pkg_version = importlib.metadata.version


def _version_skip_broken_flash(name: str):
    if name in ("flash_attn", "flash-attn"):
        raise importlib.metadata.PackageNotFoundError(name)
    return _orig_pkg_version(name)


importlib.metadata.version = _version_skip_broken_flash

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import pdb
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import argparse
import re
from PIL import Image
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random

SYSTEM_PROMPT  = '''You are a helpful assistant.'''

# step_attention: decode step 별 attention ratio viz 저장
def quick_time_avg_and_visualize_attention(
    attentions,         # attention values
    answer_end_idx,     # 생성된 답변에서 종료 토큰 위치
    pos,                # 이미지 토큰 시작 위치
    num_image_token,    # 이미지 토큰 개수
    query_end_pos,      # 질문 종료 토큰 위치
    save_folder,        # 저장 폴더
    mode,               # back, cot, default, lookback
    decoded_tokens_text_list=None,  # 생성된 토큰을 문자열로 변환한 리스트트
    output_text=None,               # 모델이 생성한 텍스트트
    processor=None                  # processor 객체체
    ):
    
    os.makedirs(save_folder, exist_ok=True)

    img_token_start = pos
    img_token_end = pos + num_image_token
    query_start_pos = img_token_end + 1
    gen_start_pos = query_end_pos + 5


    steps = answer_end_idx
    img_attentions = np.zeros(steps)
    txt_attentions = np.zeros(steps)
    gen_attentions = np.zeros(steps)

    for step in range(1, steps + 1):
        step_idx = step - 1  
        current_attentions = attentions[step]
        num_layers = len(current_attentions)

        layer_img_sum = 0.0
        layer_txt_sum = 0.0
        layer_gen_sum = 0.0

        for layer_idx in range(num_layers):
            att_matrix = current_attentions[layer_idx][0]
            att_avg = att_matrix.mean(dim=0).squeeze(0).to(torch.float32).detach().cpu().numpy()

            total_attention = att_avg[img_token_start:].sum()
            if total_attention == 0:
                continue  

            img_att = att_avg[img_token_start:img_token_end].sum() / total_attention
            txt_att = att_avg[query_start_pos:gen_start_pos].sum() / total_attention
            gen_att = att_avg[gen_start_pos:].sum() / total_attention
            layer_img_sum += img_att
            layer_txt_sum += txt_att
            layer_gen_sum += gen_att

        if num_layers > 0:
            img_attentions[step_idx] = layer_img_sum / num_layers
            txt_attentions[step_idx] = layer_txt_sum / num_layers
            gen_attentions[step_idx] = layer_gen_sum / num_layers

    back_start_positions = []
    back_end_positions = []
    if output_text and processor and decoded_tokens_text_list:


        for i, token in enumerate(decoded_tokens_text_list[:steps]):
            if token == '<':

                if (i + 2 < len(decoded_tokens_text_list) and 
                    decoded_tokens_text_list[i + 1] == 'back' and 
                    decoded_tokens_text_list[i + 2].startswith('>')):
                    back_start_positions.append(i + 1) 

            elif token == '</' or token.endswith('</'):
                if (i + 2 < len(decoded_tokens_text_list) and
                    decoded_tokens_text_list[i + 1] == 'back' and
                    decoded_tokens_text_list[i + 2].startswith('>')):
                    back_end_positions.append(i + 1)


    plt.figure(figsize=(12, 6))
    plt.plot(range(1, steps+1), img_attentions, label="Image Token", color="blue", alpha=0.8)
    plt.plot(range(1, steps+1), txt_attentions, label="Query Text Token", color="orange", alpha=0.8)
    plt.plot(range(1, steps+1), gen_attentions, label="Generated Token", color="green", alpha=0.8)
    
    for pos_mark in back_start_positions:
        if pos_mark <= steps:
            plt.axvline(x=pos_mark, color='red', linestyle='-', alpha=0.8, linewidth=2, label='<back>' if pos_mark == back_start_positions[0] else '')
    
    for pos_mark in back_end_positions:
        if pos_mark <= steps:
            plt.axvline(x=pos_mark, color='purple', linestyle='-', alpha=0.8, linewidth=2, label='</back>' if pos_mark == back_end_positions[0] else '')
    
    for start_pos, end_pos in zip(back_start_positions, back_end_positions):
        if start_pos <= steps and end_pos <= steps:
            plt.axvspan(start_pos, end_pos, alpha=0.2, color='yellow', label='<back> region' if start_pos == back_start_positions[0] else '')
    
    plt.xlabel("Generation Step")
    plt.ylabel("Attention Ratio")
    plt.title(f"Attention Distribution Over Generation Steps")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_name = f"{mode}_step_attention"
    save_path = os.path.join(save_folder, save_name)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Step-wise attention plot saved to {save_path}")

# step_attention_with_tokens, step_attention_all_tokens(decode step>50): att ratio 저장
def time_avg_and_visualize_attention(attentions, answer_end_idx, pos, num_image_token, query_end_pos, save_folder, mode, decoded_tokens_text_list, output_text=None, processor=None):
    os.makedirs(save_folder, exist_ok=True)

    img_token_start = pos
    img_token_end = pos + num_image_token
    query_start_pos = img_token_end + 1
    gen_start_pos = query_end_pos + 5

    steps = answer_end_idx
    img_attentions = np.zeros(steps)
    txt_attentions = np.zeros(steps)
    gen_attentions = np.zeros(steps)

    for step in range(1, steps + 1):
        step_idx = step - 1  
        current_attentions = attentions[step] 
        num_layers = len(current_attentions)

        layer_img_sum = 0.0
        layer_txt_sum = 0.0
        layer_gen_sum = 0.0


        for layer_idx in range(num_layers):
            att_matrix = current_attentions[layer_idx][0]
            att_avg = att_matrix.mean(dim=0).squeeze(0).to(torch.float32).detach().cpu().numpy()

            total_attention = att_avg[img_token_start:].sum()
            if total_attention == 0:
                continue 

            img_att = att_avg[img_token_start:img_token_end].sum() / total_attention
            txt_att = att_avg[query_start_pos:gen_start_pos].sum() / total_attention
            gen_att = att_avg[gen_start_pos:].sum() / total_attention
            layer_img_sum += img_att
            layer_txt_sum += txt_att
            layer_gen_sum += gen_att

        if num_layers > 0:
            img_attentions[step_idx] = layer_img_sum / num_layers
            txt_attentions[step_idx] = layer_txt_sum / num_layers
            gen_attentions[step_idx] = layer_gen_sum / num_layers

    token_labels = decoded_tokens_text_list[:steps]
    
    back_start_positions = []
    back_end_positions = []
    if output_text and processor:
        
        for i, token in enumerate(decoded_tokens_text_list[:steps]):
            if token == '<':
                if (i + 2 < len(decoded_tokens_text_list) and 
                    decoded_tokens_text_list[i + 1] == 'back' and 
                    decoded_tokens_text_list[i + 2].startswith('>')):
                    back_start_positions.append(i + 1) 
            elif token == '</' or token.endswith('</'):
                if (i + 2 < len(decoded_tokens_text_list) and
                    decoded_tokens_text_list[i + 1] == 'back' and
                    decoded_tokens_text_list[i + 2].startswith('>')):
                    back_end_positions.append(i + 1) 
        


    max_labels = 50
    if steps <= max_labels:
        step_size = 1
        xtick_positions = list(range(1, steps + 1))
        xtick_labels = token_labels
    else:
        step_size = steps // max_labels
        xtick_positions = list(range(1, steps + 1, step_size))
        if xtick_positions[-1] != steps:
            xtick_positions.append(steps)
        xtick_labels = [token_labels[i - 1] for i in xtick_positions]

    plt.figure(figsize=(15, 8))
    
    x = range(1, steps + 1)
    plt.plot(x, img_attentions, label="Image Token", color="blue", alpha=0.8)
    plt.plot(x, txt_attentions, label="Query Text Token", color="orange", alpha=0.8)
    plt.plot(x, gen_attentions, label="Generated Token", color="green", alpha=0.8)

    for pos in back_start_positions:
        if pos <= steps:
            plt.axvline(x=pos, color='red', linestyle='-', alpha=0.8, linewidth=2, label='<back>' if pos == back_start_positions[0] else '')
    
    for pos in back_end_positions:
        if pos <= steps:
            plt.axvline(x=pos, color='purple', linestyle='-', alpha=0.8, linewidth=2, label='</back>' if pos == back_end_positions[0] else '')

    for start_pos, end_pos in zip(back_start_positions, back_end_positions):
        if start_pos <= steps and end_pos <= steps:
            plt.axvspan(start_pos, end_pos, alpha=0.2, color='yellow', label='<back> region' if start_pos == back_start_positions[0] else '')
    
    plt.xticks(xtick_positions, xtick_labels, rotation=45, ha='right', fontsize=10)
    for tick in xtick_positions:
        plt.axvline(x=tick, color='gray', linestyle='--', alpha=0.3)
    
    plt.xlabel("Generated Tokens", fontsize=12)
    plt.ylabel("Attention Ratio", fontsize=12)
    plt.title(f"Attention Distribution Over Generation Steps", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    save_name = f"{mode}_step_attention_with_tokens.png"
    save_path = os.path.join(save_folder, save_name)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    if steps > max_labels:
        plt.figure(figsize=(steps / 5, 6))
        plt.plot(x, img_attentions, label="Image Token", color="blue", alpha=0.8)
        plt.plot(x, txt_attentions, label="Query Text Token", color="orange", alpha=0.8)
        plt.plot(x, gen_attentions, label="Generated Token", color="green", alpha=0.8)

        for pos in back_start_positions:
            if pos <= steps:
                plt.axvline(x=pos, color='red', linestyle='-', alpha=0.8, linewidth=2, label='<back>' if pos == back_start_positions[0] else '')
        
        for pos in back_end_positions:
            if pos <= steps:
                plt.axvline(x=pos, color='purple', linestyle='-', alpha=0.8, linewidth=2, label='</back>' if pos == back_end_positions[0] else '')

        for start_pos, end_pos in zip(back_start_positions, back_end_positions):
            if start_pos <= steps and end_pos <= steps:
                plt.axvspan(start_pos, end_pos, alpha=0.2, color='yellow', label='<back> region' if start_pos == back_start_positions[0] else '')

        plt.xticks(range(1, steps + 1), token_labels, rotation=90, ha='center', fontsize=8)

        plt.xlabel("Generated Tokens")
        plt.ylabel("Attention Ratio")
        plt.title(f"Detailed Attention Distribution (All Tokens)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        save_name_detailed = f"Step_attention_all_tokens.png"
        save_path_detailed = os.path.join(save_folder, save_name_detailed)
        plt.savefig(save_path_detailed, bbox_inches='tight')
        plt.close()

        print(f"Detailed attention plot with all tokens saved to {save_path_detailed}")

# qeury_last_token_attention_heatmap, query_last_token_attention_map_only: query_end_pos 토큰이 이미지 토큰에 주는 attention 시각화
def visualize_query_last_token_attention_on_image(
    attentions,     # attention values
    image_path,     # image path
    pos,            # 이미지 토큰 시작 위치
    num_image_token,# 이미지 토큰 개수
    query_end_pos,  # 질문 종료 토큰 위치
    output_shape,   # 이미지 토큰 그리드 형태
    save_folder,    # 저장 폴더
    mode,           # back, cot, default, lookback
    alpha=0.5       # heatmap overlay 투명도도
):
    """
    질문의 마지막 토큰(query_end_pos)이 이미지 토큰들에 걸리는 attention을
    모든 레이어·헤드에 대해 평균하여 실제 이미지 위에 패치 단위 heatmap으로 overlay한다.

    각 vision token 그리드 패치를 INTER_NEAREST resize로 선명한 직사각형 블록으로
    표현하고, 패치 경계선을 그려 첨부 이미지와 같은 격자 형태로 시각화한다.

    attentions[0]: prefill step, shape per layer → (1, num_heads, query_len, seq_len)
    """
    os.makedirs(save_folder, exist_ok=True)

    img_token_start = pos                   # image token 시작 위치
    img_token_end = pos + num_image_token   # image token 종료 위치

    prefill_attentions = attentions[0]      # tuple of tensors, one per layer. (prefill step의 attention - (1, num_heads, query_len, seq_len))
    num_layers = len(prefill_attentions)    # layer 개수

    img_att_accum = np.zeros(num_image_token, dtype=np.float64) # image token 개수만큼 0으로 초기화
    # pdb.set_trace()
    for layer_idx in range(num_layers):    # layer별 attention 계산
        # prefill_attentions: (1, num_heads, query_len, seq_len)
        att = prefill_attentions[layer_idx][0]      # (num_heads, query_len, seq_len)   # [28,808,808]
        att = att.to(torch.float32).detach().cpu()  # detach()는 gradient 계산 방지, cpu()는 tensor를 numpy로 변환 (numpy는 cpu에서 계산)

        # 각 layer에서 마지막 토큰(query_end_pos)이 각 위치에 주는 attention
        if att.dim() == 3:
            # (num_heads, query_len, seq_len)
            query_att = att[:, query_end_pos, :]  # (num_heads, seq_len)    # [28,808]
            # print("att.dim() == 3")
        else:
            # (num_heads, seq_len): single query case (decode step의 attention이 사용될 경우)
            query_att = att
            print("att.dim() == 2")

        query_att = query_att.numpy()  # (num_heads, seq_len)

        # 
        img_att = query_att[:, img_token_start:img_token_end]   # query 위치에서 image에 주는 attn. image token 범위만 추출: (num_heads, num_image_token) # [28,729]
        img_att_accum += img_att.mean(axis=0)                   # head 평균 # (729,)
        

    # pdb.set_trace()
    img_att_map = img_att_accum / num_layers    # layer 평균 (num_image_token, )    # (729,)

    # 2D reshape: output_shape = (H_grid, W_grid)
    H, W = int(output_shape[0]), int(output_shape[1])   # grid shape    # [27,27]
    img_att_map_2d = img_att_map.reshape(H, W)  # (729,) -> (27,27)

    # # 0~255 정규화 
    # att_min, att_max = img_att_map_2d.min(), img_att_map_2d.max()
    # if att_max - att_min > 1e-8:
    #     img_att_norm = (img_att_map_2d - att_min) / (att_max - att_min)
    # else:
    #     img_att_norm = np.zeros_like(img_att_map_2d)
    # img_att_uint8 = (img_att_norm * 255).astype(np.uint8) # colormap 적용 위해 0~255 정규화 (cv2.applyColorMap)

    # LogNorm 정규화 (값이 0 또는 매우 작은 경우 로그 스케일에서 문제가 생길 수 있어 하한으로 clip)
    log_vmin = 7e-4
    log_vmax = max(float(img_att_map_2d.max()), log_vmin * (1.0 + 1e-6))
    log_norm = LogNorm(vmin=log_vmin, vmax=log_vmax, clip=True)
    img_att_norm = log_norm(np.clip(img_att_map_2d, log_vmin, None))
    img_att_uint8 = (img_att_norm * 255).astype(np.uint8)

    # 원본 이미지 로드
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        orig_img_pil = Image.open(image_path).convert("RGB")
        orig_img = cv2.cvtColor(np.array(orig_img_pil), cv2.COLOR_RGB2BGR)

    orig_h, orig_w = orig_img.shape[:2]

    # INTER_NEAREST: 각 패치가 선명한 직사각형 블록으로 표현됨 (보간 없음)
    att_resized = cv2.resize(img_att_uint8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # colormap 적용
    heatmap_color = cv2.applyColorMap(att_resized, cv2.COLORMAP_JET)

    # 원본 이미지와 blend
    overlaid = cv2.addWeighted(orig_img, 1 - alpha, heatmap_color, alpha, 0)

    # 패치 경계선 그리기: 각 그리드 셀 테두리를 흰색 반투명 선으로 표시
    patch_h = orig_h / H
    patch_w = orig_w / W
    grid_overlay = overlaid.copy()
    for row in range(1, H):
        y = int(round(row * patch_h))
        cv2.line(grid_overlay, (0, y), (orig_w, y), (255, 255, 255), 1)
    for col in range(1, W):
        x = int(round(col * patch_w))
        cv2.line(grid_overlay, (x, 0), (x, orig_h), (255, 255, 255), 1)
    # 원본 overlay와 그리드 선을 0.7:0.3으로 합성하여 선이 너무 강하지 않게
    overlaid_with_grid = cv2.addWeighted(overlaid, 0.7, grid_overlay, 0.3, 0)

    save_name = f"{mode}_query_last_token_attention_heatmap.png"
    save_path = os.path.join(save_folder, save_name)
    cv2.imwrite(save_path, overlaid_with_grid)
    print(f"Query last-token attention heatmap (patch grid) saved to {save_path}")

    # attention map만 단독으로도 저장
    heatmap_only_path = os.path.join(save_folder, f"{mode}_query_last_token_attention_map_only.png")
    cv2.imwrite(heatmap_only_path, heatmap_color)
    print(f"Attention map (no overlay) saved to {heatmap_only_path}")


def process_inputs_and_generate_output(processor, messages, model, device):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(device)
    # pdb.set_trace()
    image_inputs_aux = processor.image_processor(images=image_inputs)
    output_shape = image_inputs_aux["image_grid_thw"].numpy().squeeze(0)[1:] / 2
    output_shape = output_shape.astype(int)
    num_image_token = (output_shape[0] * output_shape[1]).item()

    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')

    pos = inputs['input_ids'].tolist()[0].index(vision_start_token_id) + 1
    pos_end = inputs['input_ids'].tolist()[0].index(vision_end_token_id)
    input_list = inputs.input_ids.tolist()[0]
    sys_end_pos = input_list.index(151645)
    
    try:
        query_end_pos = input_list.index(151645, sys_end_pos + 1)
    except ValueError:
        print("query_end_pos occurrence of token 151645 not found")

    outputs = model.generate(**inputs, max_new_tokens=2048, output_attentions=True, return_dict_in_generate=True, output_hidden_states=True)
    attentions = outputs.attentions
    generated_ids = outputs.sequences

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    
    print("Generated Text:", output_text)
    decoded_tokens = processor.tokenizer.encode(output_text[0], add_special_tokens=True)
    End_idx = decoded_tokens.index(151645)
    decoded_tokens_text_list = processor.tokenizer.convert_ids_to_tokens(decoded_tokens)
    return output_text, End_idx, num_image_token, pos, query_end_pos, attentions, decoded_tokens_text_list, output_shape

def select_prompt_template(mode, question):
    """
    Function to prepend the instruction to the base text.
    """
    if mode == "back":  # <back> token
        instruct = """
        You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags, and use <back> </back> to verify your reasoning against the image. The final answer MUST BE put in \boxed{}, respectively, i.e., <think> reasoning process here </think> <back> verification process here </back> <think> continue reasoning </think> \\boxed{final answer}.
        """
    
    elif mode == "cot":
        instruct = """
        You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}, respectively, i.e., <think> reasoning process here </think> \\boxed{final answer}.
        """

    # elif mode == "back":  # <back> token x
    #     instruct = """
    #   You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think></think> tags, and during the process you must verify your reasoning against the image. The final answer MUST BE put in \boxed{}, respectively, i.e., <think> reasoning process here </think> \\boxed{final answer}.
    #     """

    elif mode == "default":
        instruct = ""

    elif mode == "lookback":
        instruct = """
        Also, verify your response against the image during generation.
        """

    return question + " " + instruct

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



if __name__ == "__main__":
    set_seed(0)

    parser = argparse.ArgumentParser(description="Run attention visualization with custom parameters")
    parser.add_argument("--mode", type=str, default="back", choices=["back", "cot", "default", "lookback"], help="Reasoning mode")
    parser.add_argument("--image_path", type=str, default="./data/images/demo11.png", help="Path to the image file")
    parser.add_argument("--question", type=str, default="Subtract 0 blue spheres. How many objects are left?", help="Question to ask")
    parser.add_argument("--model_path", type=str, default="/home/liuyuyang_2/yangshuo/attention_in_rl/MLLMs/Reflect_4", help="Path to the model directory")
    parser.add_argument("--save_folder", type=str, default="./back_qwen_output_attention_maps_new/pretrain", help="Base folder to save output files")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (cuda:0, cpu, or auto for automatic detection)")
    args = parser.parse_args()
    mode = args.mode
    image_path = args.image_path
    demo = os.path.splitext(os.path.basename(image_path))[0]
    question = args.question
    model_path = args.model_path
    save_folder = os.path.join(args.save_folder, demo)
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", attn_implementation="eager"
    )
    model = model.to(device)
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    input_text = select_prompt_template(mode, question)
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": input_text},
            ],
        }
    ]

    output_text, End_idx, num_image_token, pos, query_end_pos, attentions, decoded_tokens_text_list, output_shape = process_inputs_and_generate_output(processor, messages, model, device)
    quick_time_avg_and_visualize_attention(attentions, End_idx, pos, num_image_token, query_end_pos, save_folder, mode, decoded_tokens_text_list, output_text[0], processor)
    time_avg_and_visualize_attention(attentions, End_idx, pos, num_image_token, query_end_pos, save_folder, mode, decoded_tokens_text_list, output_text[0], processor)
    visualize_query_last_token_attention_on_image(attentions, image_path, pos, num_image_token, query_end_pos, output_shape, save_folder, mode)