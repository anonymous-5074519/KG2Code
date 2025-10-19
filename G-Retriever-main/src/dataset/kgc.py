import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import datasets
from tqdm import tqdm
from src.dataset.utils.retrieval import retrieval_via_pcst

model_name = 'sbert'
# Base dataset dir (graphs, nodes, edges, caches)
path = os.getenv('KGC_DATASET_DIR', 'dataset/kgc')
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

cached_graph = f'{path}/cached_graphs'
cached_desc = f'{path}/cached_desc'

# JSON root directory for train/dev/test json(l)
json_root = os.getenv('KGC_JSON_DIR', 'kgc')


def _available_data_files():
    def _nonempty(p):
        return os.path.exists(p) and os.path.getsize(p) > 0
    train_fp = f"{json_root}/train.jsonl"
    dev_fp = f"{json_root}/dev.jsonl"
    test_fp = f"{json_root}/test.jsonl"
    files = {}
    if _nonempty(train_fp):
        files['train'] = train_fp
    if _nonempty(dev_fp):
        files['validation'] = dev_fp
    if _nonempty(test_fp):
        files['test'] = test_fp
    if not files:
        raise FileNotFoundError(f"No non-empty splits found under {json_root}. At least test.jsonl is required.")
    return files


class KGCDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.prompt = 'Please answer the given question.'
        self.graph = None
        self.graph_type = 'Knowledge Graph'
        dataset = datasets.load_dataset(
            "json",
            data_files=_available_data_files()
        )
        concat_list = []
        for key in ['train', 'validation', 'test']:
            if key in dataset:
                concat_list.append(dataset[key])
        self.dataset = datasets.concatenate_datasets(concat_list)
        self.q_embs = torch.load(f'{path}/q_embs.pt')

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        question = f'Question: {data["question"]}\nAnswer: '
        # 若缓存缺失，在线生成一次并落盘
        if not (os.path.exists(f'{cached_graph}/{index}.pt') and os.path.exists(f'{cached_desc}/{index}.txt')):
            nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
            edges = pd.read_csv(f'{path_edges}/{index}.csv')
            graph_full = torch.load(f'{path_graphs}/{index}.pt')
            q_emb = self.q_embs[index]
            subg, desc_text = retrieval_via_pcst(graph_full, q_emb, nodes, edges, topk=3, topk_e=5, cost_e=0.5)
            os.makedirs(cached_graph, exist_ok=True)
            os.makedirs(cached_desc, exist_ok=True)
            torch.save(subg, f'{cached_graph}/{index}.pt')
            open(f'{cached_desc}/{index}.txt', 'w').write(desc_text)

        graph = torch.load(f'{cached_graph}/{index}.pt')
        # 仅GNN路径：通过环境变量跳过 desc 读取，避免无效 IO
        import os as _os
        if _os.getenv('ONLY_GNN', '0') == '1':
            desc = ''
        else:
            desc = open(f'{cached_desc}/{index}.txt', 'r').read()
        label = ('|').join(data['answer']).lower()

        return {
            'id': index,
            'question': question,
            'label': label,
            'graph': graph,
            'desc': desc,
        }

    def get_idx_split(self):
        # Guard against missing split files
        def _read_or_empty(p):
            if os.path.exists(p) and os.path.getsize(p) > 0:
                with open(p, 'r') as f:
                    return [int(line.strip()) for line in f]
            return []
        train_indices = _read_or_empty(f'{path}/split/train_indices.txt')
        val_indices = _read_or_empty(f'{path}/split/val_indices.txt')
        test_indices = _read_or_empty(f'{path}/split/test_indices.txt')
        # If train/val are empty, treat whole dataset as test (pure inference)
        if len(train_indices) + len(val_indices) + len(test_indices) == 0:
            test_indices = list(range(len(self)))
        return {'train': train_indices, 'val': val_indices, 'test': test_indices}


def preprocess():
    os.makedirs(cached_desc, exist_ok=True)
    os.makedirs(cached_graph, exist_ok=True)
    dataset = datasets.load_dataset(
        "json",
        data_files=_available_data_files()
    )

    concat_list = []
    for key in ['train', 'validation', 'test']:
        if key in dataset:
            concat_list.append(dataset[key])
    dataset = datasets.concatenate_datasets(concat_list)

    q_embs = torch.load(f'{path}/q_embs.pt')
    for index in tqdm(range(len(dataset))):
        # 只有当图与描述都存在时才跳过
        if os.path.exists(f'{cached_graph}/{index}.pt') and os.path.exists(f'{cached_desc}/{index}.txt'):
            continue

        nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        edges = pd.read_csv(f'{path_edges}/{index}.csv')
        # 不再跳过空图；retrieval_via_pcst 会自行处理
        graph = torch.load(f'{path_graphs}/{index}.pt')
        q_emb = q_embs[index]
        subg, desc = retrieval_via_pcst(graph, q_emb, nodes, edges, topk=3, topk_e=5, cost_e=0.5)
        torch.save(subg, f'{cached_graph}/{index}.pt')
        open(f'{cached_desc}/{index}.txt', 'w').write(desc)


if __name__ == '__main__':

    preprocess()

    dataset = KGCDataset()

    data = dataset[1]
    for k, v in data.items():
        print(f'{k}: {v}')

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
