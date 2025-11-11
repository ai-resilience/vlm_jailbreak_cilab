# resolutions=(560 840 1120 1400 1680)
resolutions=(560)

for resolution in ${resolutions[@]}; do
    CUDA_VISIBLE_DEVICES=3 python scripts/analysis/run_attention.py \
        --model_name qwen \
        --dataset Figstep \
        --step_indices 0 \
        --image dataset/black_imgs/qwen/blank_${resolution}.png \
        --jsonl_result /mnt/server16_hard1/kihyun/vlm_kihyun_workspace/result/evaluation/qwen/Figstep_flagged/qwen_image_Figstep_response_${resolution}_keyword.jsonl \
        --plot \
        --image_only \
        --normalize_attention \
        --top_k 10 \
        --resolution ${resolution}

done