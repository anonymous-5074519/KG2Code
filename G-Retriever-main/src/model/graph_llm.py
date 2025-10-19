import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_scatter import scatter
from src.model.gnn import load_gnn_model
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'

IGNORE_INDEX = -100


class GraphLLM(torch.nn.Module):

    def __init__(
        self,
        args,
        **kwargs
    ):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens
        # 是否仅使用GNN特征作为输入（不拼接文本）
        try:
            self.only_gnn = str(getattr(args, 'only_gnn', 'False')) == 'True'
        except Exception:
            self.only_gnn = False
        
        print('Loading LLAMA')
        # Build safe device and memory mapping
        num_visible_gpus = torch.cuda.device_count()
        use_cuda = torch.cuda.is_available() and num_visible_gpus > 0
        revision = "main"

        if use_cuda:
            # Normalize and expand max_memory list to match visible GPU count
            raw_mem = args.max_memory
            if isinstance(raw_mem, (list, tuple)):
                mem_list = [str(x).strip() for x in raw_mem]
            else:
                mem_list = [str(raw_mem).strip()]

            if len(mem_list) == 0:
                mem_list = ["24"]
            if len(mem_list) == 1 and num_visible_gpus > 1:
                mem_list = mem_list * num_visible_gpus
            elif len(mem_list) < num_visible_gpus:
                mem_list = mem_list + [mem_list[-1]] * (num_visible_gpus - len(mem_list))
            elif len(mem_list) > num_visible_gpus:
                mem_list = mem_list[:num_visible_gpus]

            def to_int_gib(x):
                try:
                    return int(float(x))
                except Exception:
                    return int(x)

            max_memory = {i: f"{to_int_gib(mem_list[i])}GiB" for i in range(num_visible_gpus)}
            kwargs = {
                "max_memory": max_memory,
                "device_map": "auto",
                "revision": revision,
            }
        else:
            # CPU-only fallback
            kwargs = {
                "device_map": "cpu",
                "revision": revision,
            }
        # print('Loading LLAMA')
        # kwargs = {
        #     "max_memory": {0: '80GiB', 1: '80GiB'},
        #     "device_map": "auto",
        #     "revision": "main",
        # }

        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"], trust_remote_code=True)
        try:
            self.tokenizer.model_max_length = int(1e9)
        except Exception:
            pass
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'
        # 若可用，启用聊天模板（避免手写指令 token 带来的不一致）
        self.use_chat_template = bool(getattr(self.tokenizer, 'apply_chat_template', None))

        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.llm_model_path,
                torch_dtype=torch.float16 if use_cuda else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                **kwargs
            )
        except RuntimeError as e:
            # Handle invalid device ordinal by forcing all weights on first visible GPU
            if "invalid device ordinal" in str(e).lower() and use_cuda:
                fallback_kwargs = {"device_map": {"": 0}, "revision": revision}
                model = AutoModelForCausalLM.from_pretrained(
                    args.llm_model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    **fallback_kwargs
                )
            else:
                raise

        # 保证模型词表与 tokenizer 长度一致（有些远程权重/模板会附带新增特殊 token）
        try:
            current_vocab = model.get_input_embeddings().num_embeddings
            target_vocab = len(self.tokenizer)
            if current_vocab != target_vocab:
                model.resize_token_embeddings(target_vocab)
        except Exception:
            pass

        if args.llm_frozen == 'True':
            print("Freezing LLAMA!")
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            print("Training LLAMA with LORA!")
            model = prepare_model_for_kbit_training(model)
            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.05
            lora_target_modules = [
                "q_proj",
                "v_proj",
            ]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

        self.model = model
        # Primary device for custom modules (works even if HF model is sharded)
        self.primary_device = next(self.model.parameters()).device
        # 同步 pad/eos 到模型配置，避免生成阶段反复提示
        try:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            if getattr(self.tokenizer, 'eos_token_id', None) is not None:
                self.model.config.eos_token_id = self.tokenizer.eos_token_id
        except Exception:
            pass
        print('Finish loading LLAMA!')

        self.graph_encoder = load_gnn_model[args.gnn_model_name](
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(self.primary_device)

        # 取得 LLM 词嵌入及隐藏维度，确保图嵌入与之对齐（兼容 2048/4096 等不同模型）
        self.word_embedding = self.model.model.get_input_embeddings()
        llm_hidden_dim = getattr(self.model.config, 'hidden_size', getattr(self.word_embedding, 'embedding_dim', None))
        if llm_hidden_dim is None:
            raise ValueError('Cannot determine LLM hidden size for projector alignment.')
        # 维持两层 MLP，但将输出维度对齐到 LLM 隐藏维度；中间层选取不超过 2048 的安全尺寸
        inter_dim = min(2048, llm_hidden_dim)
        self.projector = nn.Sequential(
            nn.Linear(args.gnn_hidden_dim, inter_dim),
            nn.Sigmoid(),
            nn.Linear(inter_dim, llm_hidden_dim),
        ).to(self.primary_device)

        # 最大序列长度，用于安全裁剪
        self.max_position_embeddings = getattr(self.model.config, 'max_position_embeddings', 4096)

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with the model's parameter dtype (fp16/bf16)
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            model_dtype = next(self.model.parameters()).dtype
            return torch.cuda.amp.autocast(dtype=model_dtype)
        else:
            return contextlib.nullcontext()

    def encode_graphs(self, samples):
        graphs = samples['graph']
        graphs = graphs.to(self.primary_device)

        num_graphs = graphs.num_graphs
        # 推断GNN输出维度（与projector输入一致）
        projector_in_dim = self.projector[0].in_features
        default_dtype = next(self.graph_encoder.parameters()).dtype

        # 若该 batch 没有任何节点（例如所有样本图均为空），直接返回零向量占位
        if graphs.x is None or graphs.x.numel() == 0:
            return torch.zeros((num_graphs, projector_in_dim), device=self.primary_device, dtype=default_dtype)

        n_embeds, _ = self.graph_encoder(graphs.x, graphs.edge_index.long(), graphs.edge_attr)

        # 对每个图做 mean pooling；dim_size 保证空图位置也会被保留（填充为0）
        g_embeds = scatter(n_embeds, graphs.batch, dim=0, dim_size=num_graphs, reduce='mean')
        # 数值稳定：把可能出现的 NaN（极端情况下 mean/0）置零
        g_embeds = torch.nan_to_num(g_embeds, nan=0.0)

        # 如果由于极端数据导致维度仍不匹配，进行安全补齐
        if g_embeds.size(0) < num_graphs:
            pad = torch.zeros((num_graphs - g_embeds.size(0), g_embeds.size(1)), device=g_embeds.device, dtype=g_embeds.dtype)
            g_embeds = torch.cat([g_embeds, pad], dim=0)

        return g_embeds

    def forward(self, samples):

        # encode description, questions and labels（仅当非only_gnn时使用文本）
        if not self.only_gnn:
            questions = self.tokenizer(samples["question"], add_special_tokens=False)
            descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)

        # encode special tokens（优先使用 tokenizer 自带的特殊 token id）
        if getattr(self.tokenizer, 'eos_token_id', None) is not None:
            eos_ids = [int(self.tokenizer.eos_token_id)]
        else:
            eos_ids = self.tokenizer(EOS, add_special_tokens=False).input_ids
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        if getattr(self.tokenizer, 'bos_token_id', None) is not None:
            bos_ids = torch.tensor([int(self.tokenizer.bos_token_id)], device=self.primary_device)
        else:
            bos_ids = self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.primary_device)
        bos_embeds = self.word_embedding(bos_ids)
        # 计算需要预留的位置：实际 BOS token 数量 + 1 个图嵌入位置
        reserved_slots = int(bos_embeds.shape[0]) + 1
        pad_id = torch.tensor(self.tokenizer.pad_token_id, device=self.primary_device)
        pad_embeds = self.word_embedding(pad_id).unsqueeze(0)

        # encode graphs
        graph_embeds = self.encode_graphs(samples)
        graph_embeds = self.projector(graph_embeds)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
            # Add bos & eos token
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_ids
            if self.only_gnn:
                # 仅GNN：不使用文本，将结构保持为 <BOS> + <GNN> + [/INST] + <label+EOS>
                input_ids = eos_user_tokens.input_ids + label_input_ids
                # 运行时快速检查：避免越界 ID 进入 embedding
                if len(input_ids) > 0:
                    max_id = max(input_ids)
                    min_id = min(input_ids)
                    vocab_size = self.word_embedding.num_embeddings
                    if max_id >= vocab_size or min_id < 0:
                        raise ValueError(f"Token id out of range in prefix/suffix: min={min_id}, max={max_id}, vocab={vocab_size}")
                inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.primary_device))
                ge = graph_embeds[i].to(bos_embeds.dtype).unsqueeze(0)
                inputs_embeds = torch.cat([bos_embeds, ge, inputs_embeds], dim=0)
            else:
                if self.use_chat_template:
                    # 使用聊天模板生成前缀（包含 system/user 等必要标记）
                    user_text = "" if self.only_gnn else (str(samples["desc"][i]) + "\n" + str(samples["question"][i]))
                    messages = [{"role": "user", "content": user_text}]
                    try:
                        prompt_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                    except TypeError:
                        prompt_ids = self.tokenizer.apply_chat_template(messages, tokenize=True)
                    # 去除可能重复的 BOS
                    if len(prompt_ids) > 0 and getattr(self.tokenizer, 'bos_token_id', None) is not None and prompt_ids[0] == self.tokenizer.bos_token_id:
                        text_prefix = prompt_ids[1:]
                    else:
                        text_prefix = prompt_ids
                else:
                    if getattr(self, 'only_gnn', False):
                        text_prefix = eos_user_tokens.input_ids
                    else:
                        text_prefix = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids

                # 动态裁剪，确保总长度不超过模型最大长度（预留 实际BOS长度 + 1 个graph 位置）
                max_ctx_len = max(0, self.max_position_embeddings - reserved_slots)
                suffix = label_input_ids
                allow_prefix = max_ctx_len - len(suffix)
                if allow_prefix <= 0:
                    # 标签优先，截取标签末尾部分
                    trimmed_suffix = suffix[-max_ctx_len:] if max_ctx_len > 0 else []
                    trimmed_prefix = []
                    input_ids = trimmed_prefix + trimmed_suffix
                else:
                    trimmed_prefix = text_prefix[-allow_prefix:]
                    input_ids = trimmed_prefix + suffix
                if len(input_ids) > 0:
                    max_id = max(input_ids)
                    min_id = min(input_ids)
                    vocab_size = self.word_embedding.num_embeddings
                    if max_id >= vocab_size or min_id < 0:
                        raise ValueError(f"Token id out of range in prefix/suffix: min={min_id}, max={max_id}, vocab={vocab_size}")
                inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.primary_device))
                # 保证图嵌入与 token 嵌入 dtype 一致，避免下游模块 dtype 冲突
                ge = graph_embeds[i].to(bos_embeds.dtype).unsqueeze(0)
                inputs_embeds = torch.cat([bos_embeds, ge, inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0]-len(label_input_ids))+label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length+batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.primary_device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.primary_device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.primary_device)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss

    def inference(self, samples):

        # encode description and questions（仅当非only_gnn时使用文本）
        if not self.only_gnn:
            questions = self.tokenizer(samples["question"], add_special_tokens=False)
            descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)

        # encode special tokens（优先使用 tokenizer 自带的特殊 token id）
        if getattr(self.tokenizer, 'bos_token_id', None) is not None:
            bos_ids = torch.tensor([int(self.tokenizer.bos_token_id)], device=self.primary_device)
        else:
            bos_ids = self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.primary_device)
        bos_embeds = self.word_embedding(bos_ids)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        # 计算需要预留的位置：实际 BOS token 数量 + 1 个图嵌入位置
        reserved_slots = int(bos_embeds.shape[0]) + 1
        pad_id = torch.tensor(self.tokenizer.pad_token_id, device=self.primary_device)
        pad_embeds = self.word_embedding(pad_id).unsqueeze(0)

        # encode graphs
        graph_embeds = self.encode_graphs(samples)
        graph_embeds = self.projector(graph_embeds)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            # Add bos & context
            if self.use_chat_template:
                user_text = "" if self.only_gnn else (str(samples["desc"][i]) + "\n" + str(samples["question"][i]))
                messages = [{"role": "user", "content": user_text}]
                try:
                    prompt_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                except TypeError:
                    prompt_ids = self.tokenizer.apply_chat_template(messages, tokenize=True)
                if len(prompt_ids) > 0 and getattr(self.tokenizer, 'bos_token_id', None) is not None and prompt_ids[0] == self.tokenizer.bos_token_id:
                    base_ids = prompt_ids[1:]
                else:
                    base_ids = prompt_ids
            else:
                if self.only_gnn:
                    base_ids = eos_user_tokens.input_ids
                else:
                    base_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids

            # 动态裁剪，确保总长度不超过模型最大长度（预留 实际BOS长度 + 1 个graph 位置）
            max_ctx_len = max(0, self.max_position_embeddings - reserved_slots)
            if len(base_ids) > max_ctx_len:
                input_ids = base_ids[-max_ctx_len:]
            else:
                input_ids = base_ids
                if len(input_ids) > 0:
                    max_id = max(input_ids)
                    min_id = min(input_ids)
                    vocab_size = self.word_embedding.num_embeddings
                    if max_id >= vocab_size or min_id < 0:
                        raise ValueError(f"Token id out of range in inference context: min={min_id}, max={max_id}, vocab={vocab_size}")
                inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.primary_device))
            inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.primary_device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.primary_device)

        # 检测是否是 DeepSeek 模型（通过 model_type 或配置）
        is_deepseek = False
        try:
            model_type = getattr(self.model.config, 'model_type', '').lower()
            is_deepseek = 'deepseek' in model_type
        except:
            pass
        
        with self.maybe_autocast():
            if is_deepseek:
                # DeepSeek 使用手动循环，避开 position_ids 兼容性问题
                batch_size = inputs_embeds.size(0)
                eos_token_id = getattr(self.tokenizer, 'eos_token_id', self.tokenizer.pad_token_id)
                generated_ids = []
                
                for step in range(self.max_new_tokens):
                    outputs_step = self.model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    next_token_logits = outputs_step.logits[:, -1, :]
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                    
                    if step == 0:
                        generated_ids = next_tokens.unsqueeze(1)
                    else:
                        generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(1)], dim=1)
                    
                    if (next_tokens == eos_token_id).all():
                        break
                    
                    next_token_embeds = self.word_embedding(next_tokens).unsqueeze(1)
                    inputs_embeds = torch.cat([inputs_embeds, next_token_embeds], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)], dim=1)
                
                outputs = generated_ids
            else:
                # Llama 等其他模型使用标准 generate()，保留 KV cache 加速
                outputs = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=self.max_new_tokens,
                    attention_mask=attention_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=getattr(self.tokenizer, 'eos_token_id', None),
                    use_cache=True  # Llama 可以正常使用缓存加速
                )
        
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {'id': samples['id'],
                'pred': pred,
                'label': samples['label'],
                'question': samples['question'],
                'desc': samples['desc'], }

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
