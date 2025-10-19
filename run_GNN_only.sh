#预处理,以kgc为例
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m src.dataset.preprocess.kgc
python -m src.dataset.kgc

#仅使用GNN，无text，kgc任务
export CUDA_VISIBLE_DEVICES=0,5 #指定卡
export ONLY_GNN=1
python train.py \
  --dataset kgc \
  --model_name graph_llm \
  --llm_model_path /path_to/Meta-Llama-3.1-8B-Instruct \
  --max_memory 24,24\
  --batch_size 1 \
  --only_gnn True \
  --eval_batch_size 1

#预处理,以kgqa为例
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m src.dataset.preprocess.kgqa
python -m src.dataset.kgqa

#仅使用GNN，无text，kgqa任务
export CUDA_VISIBLE_DEVICES=0,5 #指定卡
export ONLY_GNN=1
python train.py \
  --dataset kgqa \
  --model_name graph_llm \
  --llm_model_path /path_to/Meta-Llama-3.1-8B-Instruct \
  --max_memory 24,24\
  --batch_size 1 \
  --only_gnn True \
  --eval_batch_size 1
