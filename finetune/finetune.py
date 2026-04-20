import io
import os
from typing import List

import fire
import pickle
import numpy as np
import torch
import transformers

from model import LLM4Rec
from utils.data_utils import (
    BipartiteGraphCollator,
    BipartiteGraphDataset,
    SequentialCollator,
    SequentialDataset,
)
from utils.eval_utils import MAP_atK, MRR_atK, NDCG_atK, RecallPrecision_atK, getLabel
from utils.prompter import Prompter


def train(
    # model/data params
    base_model: str = "/home/snarayana_umass_edu/E4SRec-1/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
    data_path: str = "datasets/sequential/LastFM/",
    cache_dir: str = "",
    output_dir: str = "",
    task_type: str = "sequential",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 1,
    max_steps: int = -1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 4096,
    val_set_size: int = 0,
    lr_scheduler: str = "cosine",
    warmup_steps: int = 0,
    warmup_ratio: float = 0.1,
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # Qwen 2.5 attention projections; override from CLI for other backbones.
    lora_target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca"
    ,
    load_in_4bit: bool = True,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Params using prompt template {prompt_template_name}:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"cache_dir: {cache_dir}\n"
            f"output_dir: {output_dir}\n"
            f"task_type: {task_type}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"max_steps: {max_steps}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lr_scheduler: {lr_scheduler}\n"
            f"warmup_steps: {warmup_steps}\n"
            f"warmup_ratio: {warmup_ratio}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"load_in_4bit: {load_in_4bit}\n"
        )
    assert base_model, "Please specify a --base_model, e.g. a local Qwen snapshot path"
    if batch_size % micro_batch_size != 0:
        raise ValueError("batch_size must be divisible by micro_batch_size")
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        print(f"DDP enabled: LOCAL_RANK={os.environ.get('LOCAL_RANK')}, WORLD_SIZE={world_size}")  
        print("gradient_accumulation_steps: ", gradient_accumulation_steps)

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if task_type == 'general':
        dataset = BipartiteGraphDataset(data_path)
        user_embed, item_embed = (
            pickle.load(open(data_path + 'VanillaMF_user_embed.pkl', 'rb')),
            pickle.load(open(data_path + 'VanillaMF_item_embed.pkl', 'rb')),
        )
        item_embed = torch.cat([item_embed.mean(dim=0).unsqueeze(0), item_embed], dim=0)
        data_collator = BipartiteGraphCollator()
        input_dim = 64
    elif task_type == 'sequential':
        sasrec_embed_path = os.path.join(data_path, "SASRec_item_embed.pkl")
        if not os.path.exists(sasrec_embed_path):
            raise FileNotFoundError(
                f"Missing sequential item embeddings: {sasrec_embed_path}. "
                "Generate SASRec_item_embed.pkl before finetuning."
            )
        dataset = SequentialDataset(data_path, 50)

        class _CpuUnpickler(pickle.Unpickler):
            """Remap any CUDA storage to CPU so the file loads on CPU-only nodes."""
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
                return super().find_class(module, name)
        with open(sasrec_embed_path, 'rb') as _f:
            raw_embed = _CpuUnpickler(_f).load()  # shape: [max_raw_id, embed_dim]

        # item_map: raw_id -> new_contiguous_idx (1-based; 0 = padding)
        # Build the remapped embedding table in the same order as item_map.
        # raw_embed may be indexed by raw item id directly.
        input_dim = raw_embed.shape[1]
        n_items = len(dataset.item_map) + 1  # +1 for padding row at index 0
        item_embed = torch.zeros(n_items, input_dim)
        for raw_id, new_idx in dataset.item_map.items():
            if raw_id < raw_embed.shape[0]:
                item_embed[new_idx] = raw_embed[raw_id]

        user_embed = None
        data_collator = SequentialCollator()
    else:
        raise ValueError("task_type must be either 'general' or 'sequential'")

    model = LLM4Rec(
        base_model=base_model,
        task_type=task_type,
        cache_dir=cache_dir,
        input_dim=input_dim,
        output_dim=dataset.m_item,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
        instruction_text=prompter.generate_prompt(task_type),
        user_embeds=user_embed,
        input_embeds=item_embed,
        max_seq_len=cutoff_len,
    )

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    class PeftTrainer(transformers.Trainer):
        """Save only LoRA adapter weights in intermediate checkpoints.

        The default Trainer.save_model() serialises the full model including
        bitsandbytes quantisation buffers.  When those checkpoints are resumed
        the bnb metadata keys are unexpected in the freshly-quantised model,
        producing the 'UNEXPECTED keys' warnings and potential load errors.
        Overriding _save_checkpoint to call save_pretrained with
        save_embedding_layers=False avoids embedding/quant-buffer bloat.
        """

        def _save_checkpoint(self, model, trial, **kwargs):
            # Let the parent do book-keeping (scheduler, optimizer, rng state)
            super()._save_checkpoint(model, trial, **kwargs)
            # Re-save the model portion using PEFT's adapter-only path so that
            # the next resume does not pick up bnb quantisation buffers.
            checkpoint_folder = f"checkpoint-{self.state.global_step}"
            output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
            model.llama_model.save_pretrained(output_dir, save_embedding_layers=False)

    # Build a val dataset from the held-out valData entries when requested.
    # SequentialDataset.valData maps user -> [history, target_item]; we create
    # a flat list of (seq, label) pairs identical in format to trainData.
    eval_dataset = None
    if val_set_size > 0 and task_type == "sequential" and hasattr(dataset, "valData"):
        val_pairs = []
        for user, entry in dataset.valData.items():
            if len(entry) == 2:
                history, target = entry
                seq = history[-dataset.maxlen:]
                val_pairs.append((seq, target))
        if val_set_size < len(val_pairs):
            import random
            random.seed(42)
            val_pairs = random.sample(val_pairs, val_set_size)

        class _ValDataset(torch.utils.data.Dataset):
            def __init__(self, pairs):
                self.pairs = pairs
            def __len__(self):
                return len(self.pairs)
            def __getitem__(self, idx):
                return self.pairs[idx]

        eval_dataset = _ValDataset(val_pairs)
        print(f"Val dataset: {len(eval_dataset)} samples")

    use_val = eval_dataset is not None and len(eval_dataset) > 0
    # Align eval_steps with save_steps so best-model tracking works on small
    # datasets; use 10% of an epoch or 50 steps, whichever is smaller.
    steps_per_epoch = max(1, len(dataset) // (micro_batch_size * gradient_accumulation_steps))
    total_steps = steps_per_epoch * num_epochs if max_steps == -1 else max_steps
    computed_warmup_steps = warmup_steps if warmup_steps > 0 else max(1, int(total_steps * warmup_ratio))
    eval_save_steps = max(10, min(50, steps_per_epoch // 2))
    print(f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, warmup_steps={computed_warmup_steps}, eval_save_steps={eval_save_steps}")

    trainer = PeftTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=computed_warmup_steps,
            num_train_epochs=num_epochs,
            max_steps=max_steps,
            learning_rate=learning_rate,
            # fp16=True,
            bf16=True,
            logging_steps=1,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,         # trades compute for memory
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=0,
            eval_strategy="steps" if use_val else "no",
            save_strategy="steps",
            eval_steps=eval_save_steps if use_val else None,
            save_steps=eval_save_steps,
            lr_scheduler_type=lr_scheduler,
            output_dir=output_dir,
            save_total_limit=2,
            load_best_model_at_end=use_val,
            metric_for_best_model="eval_loss" if use_val else None,
            ddp_find_unused_parameters=False if ddp else None,
            report_to="none",
            run_name=None,
        ),
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Free optimizer states and gradient buffers before running eval inference.
    del trainer
    torch.cuda.empty_cache()

    model.eval()
    topk = [1, 5, 10, 20, 100]
    results = {'Precision': np.zeros(len(topk)),
               'Recall': np.zeros(len(topk)),
               'MRR': np.zeros(len(topk)),
               'MAP': np.zeros(len(topk)),
               'NDCG': np.zeros(len(topk))}

    testData = dataset.testData
    users = np.arange(dataset.n_user)
    with torch.inference_mode():
        for u in users:
            if task_type == 'general':
                all_pos = [dataset.allPos[u]]
                groundTruth = [testData[u]]
                inputs = torch.LongTensor([u] + all_pos[0]).cuda().unsqueeze(0)
                inputs_mask = torch.ones(inputs.shape).cuda()
                _, ratings = model.predict(inputs, inputs_mask)
                exclude_index = []
                exclude_items = []
                for range_i, its in enumerate(all_pos):
                    exclude_index.extend([range_i] * len(its))
                    exclude_items.extend(its)
                ratings[exclude_index, exclude_items] = -(1 << 10)

            elif task_type == 'sequential':
                if len(testData[u]) == 0:
                    continue
                selected_items = [[testData[u][1]] + dataset.allPos[u]]
                groundTruth = [[0]]
                device = next(model.llama_model.parameters()).device
                # Truncate to maxlen (same as training) to avoid OOM on long histories
                seq = testData[u][0][-dataset.maxlen:]
                inputs = torch.LongTensor(seq).to(device).unsqueeze(0)
                inputs_mask = torch.ones(inputs.shape, device=device)
                _, ratings = model.predict(inputs, inputs_mask)
                ratings = ratings[[[[k] * len(selected_items[0]) for k in range(len(ratings))], selected_items]]

            _, ratings_K = torch.topk(ratings, k=topk[-1])
            ratings_K = ratings_K.cpu().numpy()

            r = getLabel(groundTruth, ratings_K)
            for j, k in enumerate(topk):
                pre, rec = RecallPrecision_atK(groundTruth, r, k)
                mrr = MRR_atK(groundTruth, r, k)
                map = MAP_atK(groundTruth, r, k)
                ndcg = NDCG_atK(groundTruth, r, k)
                results['Precision'][j] += pre
                results['Recall'][j] += rec
                results['MRR'][j] += mrr
                results['MAP'][j] += map
                results['NDCG'][j] += ndcg

    for key in results.keys():
        results[key] /= float(len(users))
    print(f'Evaluation for User: \n')
    for j, k in enumerate(topk):
        print(f'Precision@{k}: {results["Precision"][j]} \n '
              f'Recall@{k}: {results["Recall"][j]} \n '
              f'MRR@{k}: {results["MRR"][j]} \n '
              f'MAP@{k}: {results["MAP"][j]} \n '
              f'NDCG@{k}: {results["NDCG"][j]} \n')

    os.makedirs(output_dir, exist_ok=True)
    # Save only the LoRA adapter weights, not the quantized base model weights.
    # Saving the full model (including bnb 4-bit buffers) causes unexpected-key
    # errors when the checkpoint is reloaded into a freshly-quantized model.
    model.llama_model.save_pretrained(output_dir, save_embedding_layers=False)
    model_path = os.path.join(output_dir, "adapter.pth")
    if task_type == 'general':
        user_proj, input_proj, score = model.user_proj.state_dict(), model.input_proj.state_dict(), model.score.state_dict()
        torch.save({'user_proj': user_proj, 'input_proj': input_proj, 'score': score}, model_path)
    elif task_type == 'sequential':
        input_proj, score = model.input_proj.state_dict(), model.score.state_dict()
        torch.save({'input_proj': input_proj, 'score': score}, model_path)
    print(f"Saved PEFT adapter to {output_dir}")
    print(f"Saved projection heads to {model_path}")


if __name__ == "__main__":
    torch.cuda.empty_cache() 
    fire.Fire(train)
