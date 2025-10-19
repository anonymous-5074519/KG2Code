import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from datasets import load_dataset, concatenate_datasets
from torch_geometric.data import Data
from src.utils.lm_modeling import load_model, load_text2embedding


model_name = 'sbert'
# Base cache dir for encoded artifacts (graphs, nodes, edges, q_embs, split)
path = os.getenv('KGC_DATASET_DIR', 'dataset/kgc')
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

# JSON root directory containing train/dev/test json(l)
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


def step_one():
    dataset = load_dataset("json", data_files=_available_data_files())

    concat_list = []
    for key in ['train', 'validation', 'test']:
        if key in dataset:
            concat_list.append(dataset[key])
    dataset = concatenate_datasets(concat_list)

    os.makedirs(path_nodes, exist_ok=True)
    os.makedirs(path_edges, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        nodes = {}
        edges = []
        for tri in dataset[i]['graph']:
            h, r, t = tri
            h = h.lower()
            t = t.lower()
            if h not in nodes:
                nodes[h] = len(nodes)
            if t not in nodes:
                nodes[t] = len(nodes)
            edges.append({'src': nodes[h], 'edge_attr': r, 'dst': nodes[t]})
        nodes = pd.DataFrame([{'node_id': v, 'node_attr': k} for k, v in nodes.items()], columns=['node_id', 'node_attr'])
        edges = pd.DataFrame(edges, columns=['src', 'edge_attr', 'dst'])

        nodes.to_csv(f'{path_nodes}/{i}.csv', index=False)
        edges.to_csv(f'{path_edges}/{i}.csv', index=False)


def generate_split():
    dataset = load_dataset("json", data_files=_available_data_files())

    len_train = len(dataset['train']) if 'train' in dataset else 0
    len_val = len(dataset['validation']) if 'validation' in dataset else 0
    len_test = len(dataset['test']) if 'test' in dataset else 0

    print("# train samples: ", len_train)
    print("# val samples: ", len_val)
    print("# test samples: ", len_test)

    os.makedirs(f'{path}/split', exist_ok=True)

    with open(f'{path}/split/train_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, np.arange(len_train))))

    with open(f'{path}/split/val_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, np.arange(len_val) + len_train)))

    with open(f'{path}/split/test_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, np.arange(len_test) + len_train + len_val)))


def step_two():
    print('Loading dataset...')
    dataset = load_dataset("json", data_files=_available_data_files())

    concat_list = []
    for key in ['train', 'validation', 'test']:
        if key in dataset:
            concat_list.append(dataset[key])
    dataset = concatenate_datasets(concat_list)
    questions = [i['question'] for i in dataset]

    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    # encode questions
    print('Encoding questions...')
    os.makedirs(path, exist_ok=True)
    q_embs = text2embedding(model, tokenizer, device, questions)
    torch.save(q_embs, f'{path}/q_embs.pt')

    print('Encoding graphs...')
    os.makedirs(path_graphs, exist_ok=True)
    for index in tqdm(range(len(dataset))):

        # nodes
        nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        edges = pd.read_csv(f'{path_edges}/{index}.csv')
        nodes.node_attr.fillna("", inplace=True)
        x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())

        # edges
        edge_attr = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
        edge_index = torch.LongTensor([edges.src.tolist(), edges.dst.tolist()])

        pyg_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes))
        torch.save(pyg_graph, f'{path_graphs}/{index}.pt')


if __name__ == '__main__':
    step_one()
    step_two()
    generate_split()
