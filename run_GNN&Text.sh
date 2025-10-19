#预处理,以kgc为例
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m src.dataset.preprocess.kgc
python -m src.dataset.kgc

#指定卡
export CUDA_VISIBLE_DEVICES=2,4,5,6,7

#标准运行，默认GNN+text
python train.py \
   --dataset kgc \
   --model_name graph_llm \
   --llm_model_path /path_to/DeepSeek-Coder-V2-Lite-Instruct \
   --max_memory 24,24,24,24,24\
   --batch_size 4 \
   --eval_batch_size 4 \
   --only_gnn False \
   --llm_frozen True \
   --llm_model_name DeepSeek-Coder-V2-Lite-Instruct

#预处理,以kgqa为例
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m src.dataset.preprocess.kgqa
python -m src.dataset.kgqa

#指定卡
export CUDA_VISIBLE_DEVICES=2,4,5,6,7

#标准运行，默认GNN+text
python train.py \
   --dataset kgqa \
   --model_name graph_llm \
   --llm_model_path /path_to/DeepSeek-Coder-V2-Lite-Instruct \
   --max_memory 24,24,24,24,24\
   --batch_size 4 \
   --eval_batch_size 4 \
   --only_gnn False \
   --llm_frozen True \
   --llm_model_name DeepSeek-Coder-V2-Lite-Instruct