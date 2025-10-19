#!/usr/bin/env bash
set -euo pipefail

# 可选：你的 conda 环境名（留空则使用当前激活环境）
ENV_NAME=${ENV_NAME:-}

# 可选：导出目录
OUT_DIR=${OUT_DIR:-$HOME/server_env}
mkdir -p "$OUT_DIR"

# 让脚本里可用 conda 命令
eval "$(conda shell.bash hook)"

# 安装 conda-pack（装在 base 即可）
conda activate base
if ! conda list -n base conda-pack >/dev/null 2>&1; then
  conda install -y conda-pack
fi

# 确定要打包的环境名；若未指定则使用当前激活环境
if [ -z "${ENV_NAME}" ]; then
  ENV_NAME="${CONDA_DEFAULT_ENV:-base}"
fi

# 激活你的可运行环境
conda activate "$ENV_NAME"

# 导出清单（用于备案/对比）
conda env export --no-builds > "$OUT_DIR/environment.yml" || true
conda list --explicit > "$OUT_DIR/conda-explicit.txt" || true
pip freeze > "$OUT_DIR/pip-freeze.txt"

# 记录 GPU/CUDA 关键信息（用于目标机对齐）
nvidia-smi > "$OUT_DIR/nvidia-smi.txt" || true
python - <<'PY' > "$OUT_DIR/torch_cuda.json"
import json, sys
info = {"python": sys.version}
try:
  import torch
  info["torch"] = torch.__version__
  info["cuda_compiled"] = getattr(torch.version, "cuda", None)
  info["cuda_available"] = torch.cuda.is_available()
  if torch.cuda.is_available():
    info["gpu_name"] = torch.cuda.get_device_name(0)
except Exception as e:
  info["torch_error"] = str(e)
print(json.dumps(info, indent=2, ensure_ascii=False))
PY

# 获取前缀路径并用前缀打包（对 base 环境更稳妥）
ENV_PREFIX="${CONDA_PREFIX:-}"
if [ -z "$ENV_PREFIX" ]; then
  echo "未检测到 CONDA_PREFIX，无法定位环境路径" >&2
  exit 1
fi

# 打包整个 conda 环境（最关键的产物）
conda pack -p "$ENV_PREFIX" -o "$OUT_DIR/${ENV_NAME}.tar.gz"

echo "Done. 打包文件在：$OUT_DIR/${ENV_NAME}.tar.gz"
echo "把整个 $OUT_DIR/ 目录拷到实验室机器即可（例如：scp -r \$USER@SERVER:$OUT_DIR ./server_env）"