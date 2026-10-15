# 멀티모달 입력 순서와 LVLM 안전성

[English README](README.md)

이 저장소는 이미지–텍스트 입력 순서가 대규모 비전-언어 모델의 안전 정렬에 미치는 영향을 연구한 실험을 재현합니다. 동일한 멀티모달 요청을 이미지가 텍스트보다 앞서는 **Image First** 조건과 텍스트가 이미지보다 앞서는 **Text First** 조건으로 평가합니다. 실험은 SafeBench (Typo), MM-SafetyBench (SD+Typo), MM-SafetyBench (SD)를 대상으로 하며, 각 벤치마크에서 500개 예제를 사용합니다.

평가 모델은 InternVL3-8B, Qwen2.5-VL-7B-Instruct, Qwen3-VL-8B-Instruct, Qwen3-VL-8B-Thinking입니다. 공격 성공률(ASR)은 Refusal, Target String, LlamaGuard3-8B, WildGuard와 논문의 앙상블 방법(EM)으로 측정합니다.

## 설치

```bash
cd experiments/input_order_effect
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

모델 가중치와 벤치마크 이미지는 포함되어 있지 않습니다. 데이터셋 준비 방법과 라이선스 참고 사항은 [DATASETS.md](docs/DATASETS.md)를 확인하십시오. 로컬 경로는 `configs/experiment.json`에서 설정하며, 모든 경로는 저장소 루트를 기준으로 한 상대 경로를 사용할 수 있습니다.

## 재현 절차

```bash
# 1. 결정론적으로 고정된 500개 예제 manifest 생성
python scripts/prepare_data.py --config configs/experiment.json

# 2. 벤치마크/모델 조합 하나 실행(다른 조합에 대해 반복하거나 스케줄러 사용)
python scripts/run_inference.py --config configs/experiment.json \
  --benchmark safebench_typo --model qwen25_vl_7b_instruct

# 3. 마지막 입력 토큰의 activation 추출
python scripts/extract_activations.py --config configs/experiment.json \
  --benchmark safebench_typo --model qwen25_vl_7b_instruct

# 4. 응답 평가
python scripts/evaluate_responses.py --config configs/experiment.json \
  --benchmark safebench_typo --model qwen25_vl_7b_instruct

# 5. 논문의 표와 figure 재현
python scripts/generate_tables.py --results-dir results/evaluations
python scripts/generate_figures.py --config configs/experiment.json
```

출력 파일의 조건명은 `image_first.json`과 `text_first.json`입니다. EM 점수는 논문의 방법을 따르며, 각 예제에 대한 Refusal, Target String, LlamaGuard 결과의 다수결로 계산합니다.

## 논문 Figure

- `figure2_wildguard_asr`: 모델과 입력 순서별 WildGuard ASR.
- `figure3_layer_cosine`: 반대 refusal direction에 대한 층별 cosine similarity.
- `figure4_pca`: harmless, harmful, attack representation의 2차원 PCA.

모델이나 데이터셋을 다운로드하지 않는 경량 검사는 `python scripts/pilot_test.py`로 실행할 수 있습니다.
