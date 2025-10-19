import os
import torch


def print_trainable_params(model):
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        num_params = param.numel()

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def _move_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _move_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(_move_to_cpu(v) for v in obj)
    return obj


def _save_checkpoint(model, optimizer, cur_epoch, args, is_best=False):
    """
    Save the checkpoint at the current epoch.
    """
    os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)

    param_grad_dic = {
        k: v.requires_grad for (k, v) in model.named_parameters()
    }
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]
    # 优先转存为 CPU 权重，规避部分 NFS/写盘问题
    # 非 best 检查点不保存优化器，进一步减小体积
    if is_best:
        try:
            optimizer_state = optimizer.state_dict()
        except Exception:
            optimizer_state = {}
    else:
        optimizer_state = {}
    save_obj = {
        "model": _move_to_cpu(state_dict),
        "optimizer": _move_to_cpu(optimizer_state),
        "config": args,
        "epoch": cur_epoch,
    }
    path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{args.seed}_checkpoint_{"best" if is_best else cur_epoch}.pth'
    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, path))
    tmp_path = path + ".tmp"
    try:
        torch.save(save_obj, tmp_path)
        os.replace(tmp_path, path)
    except Exception as e1:
        print(f"[warn] Failed to save with new zip format: {e1}. Falling back to legacy format...")
        try:
            torch.save(save_obj, tmp_path, _use_new_zipfile_serialization=False)
            os.replace(tmp_path, path)
        except Exception as e2:
            print(f"[error] Legacy save also failed: {e2}. Skip saving at epoch {cur_epoch} and continue training.")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass


def _reload_best_model(model, args):
    """
    Load the best checkpoint for evaluation.
    """
    checkpoint_path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{args.seed}_checkpoint_best.pth'

    print("Loading checkpoint from {}.".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    return model


def _reload_model(model, checkpoint_path):
    """
    Load the best checkpoint for evaluation.
    """

    print("Loading checkpoint from {}.".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    return model
