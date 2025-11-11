resolutions=(28 56 84 112 140 168 196 224 252 280 560 840 1120 1400 1680 1960 2240 2520)

for resolution in ${resolutions[@]}; do
    CUDA_VISIBLE_DEVICES=4 python scripts/analysis/run_attention.py \
        --model_name qwen \
        --dataset StrongREJECT \
        --step_indices 0 \
        --image dataset/black_imgs/qwen/blank_${resolution}.png \
        --jsonl_result result/kihyun_results/qwen/StrongREJECT/qwen_image_StrongREJECT_response_blank_${resolution}_llamaguard4_unsafe.jsonl \
        --plot \
        --image_only \
        --normalize_attention \
        --top_k 10 \
        --resolution ${resolution}

done