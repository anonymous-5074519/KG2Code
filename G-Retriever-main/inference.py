import os
import torch
import wandb
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import pandas as pd
from src.utils.seed import seed_everything
from src.config import parse_args_llama
from src.model import load_model, llama_model_path
from src.dataset import load_dataset
from src.utils.evaluate import eval_funcs
from src.utils.collate import collate_fn
from src.utils.ckpt import _reload_model, _reload_best_model


def main(args):

    # Step 1: Set up wandb
    seed = args.seed
    wandb.init(project=f"{args.project}",
               name=f"{args.dataset}_{args.model_name}_seed{seed}",
               config=args)

    seed_everything(seed=seed)
    print(args)

    dataset = load_dataset[args.dataset]()
    idx_split = dataset.get_idx_split()

    # Step 2: Build Node Classification Dataset
    test_dataset = [dataset[i] for i in idx_split['test']]
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)

    # Step 3: Build Model
    if args.llm_model_path:
        print(f'Using custom llm_model_path: {args.llm_model_path}')
    else:
        args.llm_model_path = llama_model_path[args.llm_model_name]
    model = load_model[args.model_name](graph=dataset.graph, graph_type=dataset.graph_type, args=args)

    # Step 3.1: Load checkpoint if provided (evaluation-only)
    if getattr(args, 'ckpt_path', ''):
        model = _reload_model(model, args.ckpt_path)
        print(f"Loaded checkpoint from {args.ckpt_path}")
    else:
        # Fallback: load the best checkpoint following training naming rule if exists
        try:
            model = _reload_best_model(model, args)
            print("Loaded best checkpoint by default naming rule.")
        except Exception as e:
            print(f"Warning: best checkpoint not found with default naming rule ({e}). Proceeding with current weights.")

    # Step 4. Evaluating
    os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)
    path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{seed}.csv'
    print(f'path: {path}')

    model.eval()
    progress_bar_test = tqdm(range(len(test_loader)))
    with open(path, "w") as f:
        for _, batch in enumerate(test_loader):
            with torch.no_grad():
                output = model.inference(batch)
                df = pd.DataFrame(output)
                for _, row in df.iterrows():
                    f.write(json.dumps(dict(row)) + "\n")
            progress_bar_test.update(1)

    # Step 5. Post-processing & Evaluating
    acc = eval_funcs[args.dataset](path)
    print(f'Test Acc {acc}')
    wandb.log({'Test Acc': acc})


if __name__ == "__main__":

    args = parse_args_llama()

    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
