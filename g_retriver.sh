conda create -n g_retriever python=3.8 -y
conda activate g_retriever

# 可选：升级 pip（不影响包版本复现，只是安装更稳）
python -m pip install -U pip

# PyTorch + cu118
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# PyG 编译扩展（严格匹配 torch 2.0.0 + cu118）
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

pip install torch-geometric==2.6.1

pip install \
  transformers==4.46.3 accelerate==1.0.1 peft==0.13.2 \
  sentencepiece==0.2.0 datasets==3.1.0 ogb==1.3.6 \
  pandas==2.0.3 scipy==1.10.1 gensim==4.3.3 \
  protobuf==4.22.1 pcst_fast==1.0.10 scikit-learn==1.3.2 \
  wandb==0.21.4 tokenizers==0.20.3 safetensors==0.5.3 \
  huggingface-hub==0.34.4 pyarrow==17.0.0