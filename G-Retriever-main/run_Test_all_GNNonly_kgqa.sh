#用于自动化遍历测试最佳权重模型在不同kgqa测试数据的表现，此脚本用于测试仅仅GNN版本，要修改为标准版本（GNN+Text)只需将下文脚本参数改为--only_gnn False即可，或者参考run_Test_all_kgqa.sh
#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5
export ONLY_GNN=1

# 两个模型的路径与权重（按需修改）
LLAMA_MODEL_PATH="/path_to/Meta-Llama-3.1-8B-Instruct"
LLAMA_CKPT="/path_to/output/kgqa/model_name_graph_llm_llm_model_name_Meta-Llama-3.1-8B-Instruct-GNN_llm_frozen_True_max_txt_len_512_max_new_tokens_32_gnn_model_name_gt_patience_2_num_epochs_10_seed0_checkpoint_best.pth"

DEEPSEEK_MODEL_PATH="/path_to/DeepSeek-Coder-V2-Lite-Instruct"
DEEPSEEK_CKPT="/path_to/output/kgqa/model_name_graph_llm_llm_model_name_DeepSeek-Coder-V2-Lite-Instruct-GNN_llm_frozen_True_max_txt_len_512_max_new_tokens_32_gnn_model_name_gt_patience_2_num_epochs_10_seed0_checkpoint_best.pth"

# 需要评测的文件（放在 Test/ 下）；直接在这里增删即可
FILES=("QALD-9.jsonl" "QALD-10.jsonl" "test.json")

# 只跑 Llama 的文件（用文件stem名，不含扩展名）；默认 test 只跑 Llama
SKIP_DEEPSEEK_FILES=("test")

contains() { # contains "needle" in "${haystack[@]}"
  local needle="$1"; shift
  for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done
  return 1
}

run_one() {
  local test_src="$1"                   # Test/xxx.jsonl|json
  local stem="${test_src%.*}"           # 去扩展名
  local json_root="tmp_json/${stem}"    # 临时 JSON 根目录
  local cache_root="dataset/Test_${stem}"     # 独立缓存根目录（不会污染 dataset/kgqa）
  local out_dir="output/Test/${stem}"         # 输出目录

  mkdir -p "$json_root" "$out_dir"
  # 统一命名为 test.jsonl；若源是 .json 也直接复制（datasets能识别JSON/JSONL）
  cp "Test/${test_src}" "${json_root}/test.jsonl"
  :> "${json_root}/train.jsonl"
  :> "${json_root}/dev.jsonl"

  # 预处理到独立缓存目录
  export KGQA_JSON_DIR="$json_root"
  export KGQA_DATASET_DIR="$cache_root"
  python -m src.dataset.preprocess.kgqa
  python -m src.dataset.kgqa


  # Llama 推理（快，使用 generate+cache）
  python inference.py \
    --dataset kgqa \
    --model_name graph_llm \
    --llm_model_path "${LLAMA_MODEL_PATH}" \
    --max_memory 24,24\
    --eval_batch_size 32 \
    --only_gnn True \
    --llm_frozen True \
    --llm_model_name Meta-Llama-3.1-8B-Instruct-GNN \
    --ckpt_path "${LLAMA_CKPT}" \
    --output_dir "${out_dir}"

  # DeepSeek 推理（慢，内部已自动走手动循环）
  if ! contains "${stem}" "${SKIP_DEEPSEEK_FILES[@]}"; then
    python inference.py \
      --dataset kgqa \
      --model_name graph_llm \
      --llm_model_path "${DEEPSEEK_MODEL_PATH}" \
      --max_memory 24,24\
      --only_gnn True \
      --llm_frozen True \
      --llm_model_name DeepSeek-Coder-V2-Lite-Instruct-GNN \
      --ckpt_path "${DEEPSEEK_CKPT}" \
      --output_dir "${out_dir}"
  fi
}

for f in "${FILES[@]}"; do
  run_one "$f"
done

echo "全部完成。结果见 output/Test/<文件stem>/kgqa/*.csv"