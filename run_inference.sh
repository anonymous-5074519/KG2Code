#用于测评最佳权重的结果并得到对应csv文件
#Deepseek kgqa gnn only
export CUDA_VISIBLE_DEVICES=4,5
export ONLY_GNN=1
python inference.py \
  --dataset kgqa \
  --model_name graph_llm \
  --max_memory 24,24\
  --llm_model_path /path_to/DeepSeek-Coder-V2-Lite-Instruct \
  --eval_batch_size 32 \
  --only_gnn True \
  --llm_frozen True \
  --llm_model_name DeepSeek-Coder-V2-Lite-Instruct-GNN \
  --ckpt_path /path_to/output/kgqa/model_name_graph_llm_llm_model_name_DeepSeek-Coder-V2-Lite-Instruct-GNN_llm_frozen_True_max_txt_len_512_max_new_tokens_32_gnn_model_name_gt_patience_2_num_epochs_10_seed0_checkpoint_best.pth

#用于测评最佳权重的结果并得到对应csv文件
# kgqa gNN+text
export CUDA_VISIBLE_DEVICES=4,5,6,7
python inference.py \
  --dataset kgqa \
  --model_name graph_llm \
  --max_memory 24,24,24,24\
  --llm_model_path /path_to/Meta-Llama-3.1-8B-Instruct \
  --eval_batch_size 12 \
  --only_gnn False \
  --llm_frozen True \
  --llm_model_name Meta-Llama-3.1-8B-Instruct \
  --ckpt_path /path_to/output/kgqa/model_name_graph_llm_Llama-3.1-8b_llm_frozen_True_max_txt_len_512_max_new_tokens_32_gnn_model_name_gt_patience_2_num_epochs_10_seed0_checkpoint_best.pth

