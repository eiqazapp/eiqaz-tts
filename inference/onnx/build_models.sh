#!/usr/bin/env bash
# بناء نماذج ONNX لتجربة المتصفح (Linux/macOS للتطوير)
# Windows: استخدم build_models.bat
set -e
cd "$(dirname "$0")"
echo "[1/4] FP32 + parity ..."
python export_onnx.py --checkpoint states_sanity_random.pth "$@"
echo "[2/4] التكميم ..."
python quantize_onnx.py
echo "[3/4] بيانات خط النص ..."
python gen_textpipe_data.py
echo "[4/4] قياس الخادم ..."
python bench_onnx.py
echo "تم — شغّل web-exp/serve.py"
