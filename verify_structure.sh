#!/bin/bash
# 구조 검증 스크립트

echo "🔍 VLM Refactoring 구조 확인"
echo "========================================"
echo ""

# 핵심 디렉토리 체크
echo "📂 주요 디렉토리..."
for dir in src scripts configs dataset eval result tests utils; do
    if [ -d "$dir" ]; then
        echo "  ✓ $dir/"
    else
        echo "  ✗ $dir/ (없음)"
    fi
done
echo ""

# src 모듈 체크
echo "📦 src 모듈..."
for module in models datasets analysis hooks inference; do
    if [ -d "src/$module" ]; then
        echo "  ✓ src/$module/"
    else
        echo "  ✗ src/$module/ (없음)"
    fi
done
echo ""

# scripts 서브디렉토리 체크
echo "📜 scripts..."
for subdir in inference analysis eval; do
    if [ -d "scripts/$subdir" ]; then
        echo "  ✓ scripts/$subdir/"
    else
        echo "  ✗ scripts/$subdir/ (없음)"
    fi
done
echo ""

# 주요 파일 체크
echo "📄 주요 파일..."
for file in README.md USAGE.md MIGRATION.md PROJECT_SUMMARY.md requirements.txt setup.py .gitignore; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (없음)"
    fi
done
echo ""

# 설정 파일 체크
echo "⚙️  설정 파일..."
for config in models.yaml datasets.yaml default.yaml; do
    if [ -f "configs/$config" ]; then
        echo "  ✓ configs/$config"
    else
        echo "  ✗ configs/$config (없음)"
    fi
done
echo ""

# Python 파일 개수
echo "🐍 Python 파일:"
py_count=$(find src scripts -name "*.py" 2>/dev/null | wc -l)
echo "  총 $py_count 개"
echo ""

# 심볼릭 링크 체크
echo "🔗 심볼릭 링크..."
if [ -L "dataset" ]; then
    echo "  ✓ dataset -> $(readlink dataset)"
else
    if [ -d "dataset" ]; then
        echo "  ℹ dataset (디렉토리 존재)"
    else
        echo "  ✗ dataset (없음)"
    fi
fi
if [ -L "eval" ]; then
    echo "  ✓ eval -> $(readlink eval)"
else
    if [ -d "eval" ]; then
        echo "  ℹ eval (디렉토리 존재)"
    else
        echo "  ✗ eval (없음)"
    fi
fi
echo ""

# Python import 테스트
echo "🔬 Python import 테스트..."
python3 -c "import sys; sys.path.insert(0, '.'); from src.models import load_model" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  ✓ src.models"
else
    echo "  ✗ src.models (import 실패)"
fi

python3 -c "import sys; sys.path.insert(0, '.'); from src.datasets import load_dataset" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  ✓ src.datasets"
else
    echo "  ✗ src.datasets (import 실패)"
fi

python3 -c "import sys; sys.path.insert(0, '.'); from src.analysis import pca_basic" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  ✓ src.analysis"
else
    echo "  ✗ src.analysis (import 실패)"
fi

python3 -c "import sys; sys.path.insert(0, '.'); from src.hooks import HookManager" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  ✓ src.hooks"
else
    echo "  ✗ src.hooks (import 실패)"
fi

python3 -c "import sys; sys.path.insert(0, '.'); from src.inference import generate_response" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  ✓ src.inference"
else
    echo "  ✗ src.inference (import 실패)"
fi
echo ""

echo "========================================"
echo "✅ 구조 확인 완료!"
echo ""
echo "다음 단계:"
echo "  1. README.md 확인"
echo "  2. USAGE.md 사용법 확인"
echo "  3. pip install -r requirements.txt 설치"
echo "  4. python scripts/inference/run_inference.py --help 테스트"

