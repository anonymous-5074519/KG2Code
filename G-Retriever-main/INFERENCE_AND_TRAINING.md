### G-Retriever 推理与训练快速上手

本文档汇总仓库中的关键脚本，带你以最短路径跑通：数据预处理 → 冻结 LLM 训练（GNN+Text 与仅 GNN）→ 推理导出 CSV；并说明自动化测试脚本的使用与可调参数。

---

### 快速运行（以 KGQA + GNN+Text 为例）

```bash
bash g_retriver.sh
python -m src.dataset.preprocess.kgqa && python -m src.dataset.kgqa
export CUDA_VISIBLE_DEVICES=0
python train.py --dataset kgqa --model_name graph_llm --llm_model_path /path/to/Meta-Llama-3.1-8B-Instruct --max_memory 24 --batch_size 4 --eval_batch_size 4 --only_gnn False --llm_frozen True --llm_model_name Meta-Llama-3.1-8B-Instruct
python inference.py --dataset kgqa --model_name graph_llm --max_memory 24 --llm_model_path /path/to/Meta-Llama-3.1-8B-Instruct --eval_batch_size 12 --only_gnn False --llm_frozen True --llm_model_name Meta-Llama-3.1-8B-Instruct --ckpt_path /path/to/best.pth
```

仅 GNN 请见下文“训练：仅 GNN”。若切换到 KGC，将 `--dataset kgqa` 改为 `--dataset kgc`，并替换 `--ckpt_path` 与 `--llm_model_name`（见参数对照）。

---

### 1. 环境准备与约定

- 建议在 Linux/WSL2 下运行 `.sh`；Windows 原生命令行不支持 bash。
- 安装依赖：

```bash
bash g_retriver.sh
```

- 模型目录 `--llm_model_path`
  - 例：`/path/to/Meta-Llama-3.1-8B-Instruct`、`/path/to/DeepSeek-Coder-V2-Lite-Instruct`
- GPU 指定：`export CUDA_VISIBLE_DEVICES=0,1,...`；`--max_memory` 需与 GPU 数量一致（逗号分隔，每个数字为对应 GPU 可用显存上限，单位 GB）。

---

### 2. 数据预处理

仓库已在脚本中给出标准预处理流程。最短命令如下：

- KGC

```bash
python -m src.dataset.preprocess.kgc
python -m src.dataset.kgc
```

- KGQA

```bash
python -m src.dataset.preprocess.kgqa
python -m src.dataset.kgqa
```

可选：通过环境变量定向数据/缓存目录（与自动化测试脚本一致的约定）。

```bash
# KGC 自定义
export KGC_JSON_DIR=tmp_json/your_set
export KGC_DATASET_DIR=dataset/Test_your_set

# KGQA 自定义
export KGQA_JSON_DIR=tmp_json/your_set
export KGQA_DATASET_DIR=dataset/Test_your_set
```

---

### 3. 训练（冻结 LLM）

下面命令直接对应仓库脚本：`run_GNN_only.sh`（仅 GNN）与 `run_GNN&Text.sh`（GNN+Text）。将 `/path/to/...` 改为你的本地模型路径。

#### 3.1 仅 GNN（不使用文本）

- KGC

```bash
export CUDA_VISIBLE_DEVICES=0
export ONLY_GNN=1
python train.py \
  --dataset kgc \
  --model_name graph_llm \
  --llm_model_path /path/to/Meta-Llama-3.1-8B-Instruct \
  --max_memory 24 \
  --batch_size 1 \
  --only_gnn True \
  --eval_batch_size 1
```

- KGQA

```bash
export CUDA_VISIBLE_DEVICES=0
export ONLY_GNN=1
python train.py \
  --dataset kgqa \
  --model_name graph_llm \
  --llm_model_path /path/to/Meta-Llama-3.1-8B-Instruct \
  --max_memory 24 \
  --batch_size 1 \
  --only_gnn True \
  --eval_batch_size 1
```

> 说明：仅 GNN 时仍需提供 `--llm_model_path`，但推理/训练中不会启用文本侧；最佳权重文件名后缀通常含有 `-GNN`。

#### 3.2 GNN + Text（推荐）

- KGC（DeepSeek 例）

```bash
export CUDA_VISIBLE_DEVICES=0,1
python train.py \
  --dataset kgc \
  --model_name graph_llm \
  --llm_model_path /path/to/DeepSeek-Coder-V2-Lite-Instruct \
  --max_memory 24,24 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --only_gnn False \
  --llm_frozen True \
  --llm_model_name DeepSeek-Coder-V2-Lite-Instruct
```

- KGQA（Llama 例）

```bash
export CUDA_VISIBLE_DEVICES=0,1
python train.py \
  --dataset kgqa \
  --model_name graph_llm \
  --llm_model_path /path/to/Meta-Llama-3.1-8B-Instruct \
  --max_memory 24,24 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --only_gnn False \
  --llm_frozen True \
  --llm_model_name Meta-Llama-3.1-8B-Instruct
```

---

### 4. 推理与导出 CSV

参考 `run_inference.sh`，评测最佳权重并在 `output/...` 生成对应 CSV。

#### 4.1 KGQA

- 仅 GNN（DeepSeek 例）

```bash
export CUDA_VISIBLE_DEVICES=0
export ONLY_GNN=1
python inference.py \
  --dataset kgqa \
  --model_name graph_llm \
  --max_memory 24 \
  --llm_model_path /path/to/DeepSeek-Coder-V2-Lite-Instruct \
  --eval_batch_size 32 \
  --only_gnn True \
  --llm_frozen True \
  --llm_model_name DeepSeek-Coder-V2-Lite-Instruct-GNN \
  --ckpt_path /path/to/best_gnn_ckpt.pth
```

- GNN+Text（Llama 例）

```bash
export CUDA_VISIBLE_DEVICES=0,1
python inference.py \
  --dataset kgqa \
  --model_name graph_llm \
  --max_memory 24,24 \
  --llm_model_path /path/to/Meta-Llama-3.1-8B-Instruct \
  --eval_batch_size 12 \
  --only_gnn False \
  --llm_frozen True \
  --llm_model_name Meta-Llama-3.1-8B-Instruct \
  --ckpt_path /path/to/best_llama_ckpt.pth
```

#### 4.2 切换到 KGC 的最小改动

- 将 `--dataset kgqa` 改为 `--dataset kgc`；
- 选择匹配的数据集权重路径 `--ckpt_path`；
- 仅 GNN：`--llm_model_name` 使用带 `-GNN` 的名称（如 `Meta-Llama-3.1-8B-Instruct-GNN` / `DeepSeek-Coder-V2-Lite-Instruct-GNN`）。

> 可选参数：`--output_dir` 指定输出目录；默认在 `output/<task>/...`。

---

### 5. 自动化测试脚本（批量评测）

脚本位于仓库根目录：

- `run_Test_all_GNNonly_kgc.sh`：KGC 仅 GNN 批量测试
- `run_Test_all_GNNonly_kgqa.sh`：KGQA 仅 GNN 批量测试
- `run_Test_all_kgqa.sh`：KGQA GNN+Text 批量测试

基本用法（任选一条）：

```bash
bash run_Test_all_GNNonly_kgc.sh
bash run_Test_all_GNNonly_kgqa.sh
bash run_Test_all_kgqa.sh
```

常改参数（脚本顶部）：

```bash
# 模型与权重（务必改为你自己的路径）
LLAMA_MODEL_PATH="/path/to/Meta-Llama-3.1-8B-Instruct"
LLAMA_CKPT="/path/to/llama_best.pth"
DEEPSEEK_MODEL_PATH="/path/to/DeepSeek-Coder-V2-Lite-Instruct"
DEEPSEEK_CKPT="/path/to/deepseek_best.pth"

# 需要评测的集合（位于 Test/ 目录）
FILES=("QALD-9.jsonl" "QALD-10.jsonl" "test.json")

# DeepSeek 跳过名单（按 stem 名，即文件名去扩展名）
SKIP_DEEPSEEK_FILES=("test")

# 显卡与批量
export CUDA_VISIBLE_DEVICES=0,1
```

扩展评测集：将你的 `xxx.jsonl/json` 放入 `Test/`，把文件名加入 `FILES=(...)`。脚本会自动：

1) 复制为 `tmp_json/<stem>/test.jsonl` 并生成空的 `train/dev`；

2) 预处理到独立缓存目录 `dataset/Test_<stem>`，不污染正式数据集缓存；

3) 在 `output/Test/<stem>/<task>/*.csv` 产出结果。

---

### 6. 参数对照与说明

| 参数 | 含义 | 常见取值/示例 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | 指定使用的 GPU 序号 | `0` 或 `0,1,2` |
| `--max_memory` | 按 GPU 顺序的显存上限（GB） | 单卡 `24`；双卡 `24,24` |
| `--dataset` | 任务/数据集 | `kgqa`、`kgc` |
| `--only_gnn` | 是否仅用 GNN | `True`/`False` |
| `--llm_frozen` | 冻结 LLM 参数 | 一般为 `True` |
| `--llm_model_path` | LLM 本地权重目录 | `/path/to/Meta-Llama-3.1-8B-Instruct` |
| `--llm_model_name` | LLM 名称标识（含/不含 `-GNN`） | `Meta-Llama-3.1-8B-Instruct`、`DeepSeek-Coder-V2-Lite-Instruct(-GNN)` |
| `--ckpt_path` | 训练生成的最佳权重 | `/path/to/...checkpoint_best.pth` |
| `--batch_size` | 训练批量 | 依据显存调整 |
| `--eval_batch_size` | 推理批量 | 依据显存与模型大小调整 |
| `--output_dir` | 推理结果输出目录 | 不指定则用默认结构 |

小贴士：

- `--max_memory` 的逗号个数要与 `CUDA_VISIBLE_DEVICES` 中 GPU 数一致；
- OOM：减小 `batch_size`/`eval_batch_size` 或减少使用的 GPU 数；
- 权重/模型路径不一致会导致加载报错，请核对 `--llm_model_name` 与权重的命名是否匹配（是否包含 `-GNN`）。

---

### 7. 常见问题（FAQ）

- Windows 如何运行？建议安装 WSL2（Ubuntu）或在远程 Linux 服务器上运行；`.sh` 需在 bash 中执行。
- CSV 在哪里？
  - 单次推理：若未指定 `--output_dir`，将按默认结构保存在 `output/<task>/...`，自动命名包含模型与参数信息；
  - 批量脚本：在 `output/Test/<stem>/<task>/*.csv`。
- 如何快速切换到 KGC？把文中的 `kgqa` 改为 `kgc`，并替换相应 `--ckpt_path` 与 `--llm_model_name`（是否含 `-GNN`）。

---

### 8. 相关脚本清单（便于对照）

- 训练：`run_GNN_only.sh`、`run_GNN&Text.sh`
- 推理：`run_inference.sh`
- 批量评测：`run_Test_all_GNNonly_kgc.sh`、`run_Test_all_GNNonly_kgqa.sh`、`run_Test_all_kgqa.sh`

以上即为从 0 到可复现实验的最小指引。祝你实验顺利！


