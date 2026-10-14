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
import numpy as np
import argparse
import re
from PIL import Image
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable

SYSTEM_PROMPT  = '''You are a helpful assistant.'''


def quick_time_avg_and_visualize_attention(attentions, answer_end_idx, pos, num_image_token, query_end_pos, save_folder, mode, decoded_tokens_text_list=None, output_text=None, processor=None):
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

#     elif mode == "back":  # <back> token x
#         instruct = """
# You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think></think> tags, and during the process you must verify your reasoning against the image. The final answer MUST BE put in \boxed{}, respectively, i.e., <think> reasoning process here </think> \\boxed{final answer}.
#         """

    elif mode == "default":
        instruct = ""

    elif mode == "lookback":
        instruct = """
        Also, verify your response against the image during generation.
        """

    return question + " " + instruct


if __name__ == "__main__":
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