import io
import os
import json
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
    base_model: str = "/work/pi_dagarwal_umass_edu/project_7/srikar/dolby-research/models/qwen2.5-7b-instruct",
    data_path: str = "datasets/sequential/LastFM/",
    embed_pkl: str = "",  # override embedding pkl path; if empty uses SASRec_item_embed.pkl
    cache_dir: str = "",
    output_dir: str = "",
    task_type: str = "sequential",
    # proj_warmup: LLM + LoRA frozen, only input_proj + score train (high LR)
    # lora_finetune: LoRA unfrozen, joint training of projection + LLM adapters (low LR)
    proj_warmup_epochs: int = 1,
    proj_warmup_lr: float = 1e-3,
    lora_finetune_epochs: int = 4,
    lora_finetune_lr: float = 5e-5,
    # legacy single-phase params (used when proj_warmup_epochs=0)
    num_epochs: int = 0,
    learning_rate: float = 5e-5,
    max_steps: int = -1,
    # batch params
    batch_size: int = 64,
    micro_batch_size: int = 2,
    cutoff_len: int = 128,
    val_set_size: int = 500,
    lr_scheduler: str = "cosine",
    warmup_steps: int = 0,
    warmup_ratio: float = 0.05,
    # lora hyperparams
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # llm hyperparams
    train_on_inputs: bool = False,
    add_eos_token: bool = False,
    group_by_length: bool = False,
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",
    wandb_log_model: str = "",
    resume_from_checkpoint: str = None,
    prompt_template_name: str = "alpaca",
    load_in_4bit: bool = True,
):
    # Type A (sasrec_only): num_epochs provided, no proj_warmup, single-phase LoRA.
    # Type B (multimodal): proj_warmup_epochs + lora_finetune_epochs, two-phase.
    if num_epochs > 0:
        proj_warmup_epochs = 0
        lora_finetune_epochs = num_epochs
        lora_finetune_lr = learning_rate

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Params using prompt template {prompt_template_name}:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"embed_pkl: {embed_pkl}\n"
            f"output_dir: {output_dir}\n"
            f"task_type: {task_type}\n"
            f"proj_warmup_epochs: {proj_warmup_epochs}, proj_warmup_lr: {proj_warmup_lr}\n"
            f"lora_finetune_epochs: {lora_finetune_epochs}, lora_finetune_lr: {lora_finetune_lr}\n"
            f"batch_size: {batch_size}, micro_batch_size: {micro_batch_size}\n"
            f"max_steps: {max_steps}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lr_scheduler: {lr_scheduler}\n"
            f"warmup_ratio: {warmup_ratio}\n"
            f"lora_r: {lora_r}, lora_alpha: {lora_alpha}, lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"load_in_4bit: {load_in_4bit}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        )

    assert base_model, "Please specify a --base_model"
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

    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # ── Load dataset + embeddings ─────────────────────────────────────────────
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
        sasrec_embed_path = embed_pkl if embed_pkl else os.path.join(data_path, "SASRec_item_embed.pkl")
        if not os.path.exists(sasrec_embed_path):
            raise FileNotFoundError(
                f"Missing sequential item embeddings: {sasrec_embed_path}. "
                "Run build_multimodal_embed.py before finetuning."
            )
        dataset = SequentialDataset(data_path, 50)

        class _CpuUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
                return super().find_class(module, name)

        with open(sasrec_embed_path, 'rb') as _f:
            raw_embed = _CpuUnpickler(_f).load()

        input_dim = raw_embed.shape[1]
        n_items = len(dataset.item_map) + 1
        item_embed = torch.zeros(n_items, input_dim)
        for raw_id, new_idx in dataset.item_map.items():
            if raw_id < raw_embed.shape[0]:
                item_embed[new_idx] = raw_embed[raw_id]

        user_embed = None
        data_collator = SequentialCollator()
    else:
        raise ValueError("task_type must be either 'general' or 'sequential'")

    # ── Build model ───────────────────────────────────────────────────────────
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
        model.is_parallelizable = True
        model.model_parallel = True

    # ── Val dataset ───────────────────────────────────────────────────────────
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
    steps_per_epoch = max(1, len(dataset) // (micro_batch_size * gradient_accumulation_steps))

    # ── PeftTrainer: correct checkpoint save + load ───────────────────────────
    class PeftTrainer(transformers.Trainer):
        """Save only LoRA adapter weights; restore them correctly on resume."""

        def _save_checkpoint(self, model, trial, **kwargs):
            super()._save_checkpoint(model, trial, **kwargs)
            # Use the same folder name super() just wrote to
            checkpoint_folder = f"checkpoint-{self.state.global_step}"
            ckpt_dir = os.path.join(self.args.output_dir, checkpoint_folder)
            if not os.path.isdir(ckpt_dir):
                return  # super() didn't actually save (e.g. save_steps not reached)
            model.llama_model.save_pretrained(ckpt_dir, save_embedding_layers=False)
            proj_path = os.path.join(ckpt_dir, "projections.pth")
            state = {"input_proj": model.input_proj.state_dict(),
                     "score": model.score.state_dict()}
            if hasattr(model, "user_proj"):
                state["user_proj"] = model.user_proj.state_dict()
            torch.save(state, proj_path)

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            from peft import set_peft_model_state_dict
            import safetensors.torch as sf
            import logging
            logger = logging.getLogger(__name__)

            if model is None:
                model = self.model

            # Load LoRA adapter weights
            adapter_path = os.path.join(resume_from_checkpoint, "adapter_model.safetensors")
            bin_path = os.path.join(resume_from_checkpoint, "adapter_model.bin")
            if os.path.exists(adapter_path):
                adapter_state = sf.load_file(adapter_path, device="cpu")
                set_peft_model_state_dict(model.llama_model, adapter_state, adapter_name="default")
                logger.info(f"Loaded LoRA adapter from {adapter_path}")
            elif os.path.exists(bin_path):
                adapter_state = torch.load(bin_path, map_location="cpu", weights_only=False)
                set_peft_model_state_dict(model.llama_model, adapter_state, adapter_name="default")
                logger.info(f"Loaded LoRA adapter from {bin_path}")
            else:
                logger.warning(f"No adapter weights found in {resume_from_checkpoint}")

            # Load projection heads
            proj_path = os.path.join(resume_from_checkpoint, "projections.pth")
            if os.path.exists(proj_path):
                proj_state = torch.load(proj_path, map_location="cpu", weights_only=False)
                model.input_proj.load_state_dict(proj_state["input_proj"])
                model.score.load_state_dict(proj_state["score"])
                if "user_proj" in proj_state and hasattr(model, "user_proj"):
                    model.user_proj.load_state_dict(proj_state["user_proj"])
                logger.info(f"Loaded projection heads from {proj_path}")

    # ── Helper to build TrainingArguments ─────────────────────────────────────
    # Type A: single-phase, checkpoints go to output_dir directly.
    # Type B: two-phase, each phase gets its own subdir.
    single_phase = proj_warmup_epochs == 0

    def _make_training_args(n_epochs, lr, total_steps, phase):
        computed_warmup = warmup_steps if warmup_steps > 0 else max(1, int(total_steps * warmup_ratio))
        eval_save_steps = max(10, min(50, steps_per_epoch // 2))
        phase_dir = output_dir if single_phase else os.path.join(output_dir, phase)
        return transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=computed_warmup,
            num_train_epochs=n_epochs,
            max_steps=max_steps if phase == "lora_finetune" else -1,
            learning_rate=lr,
            bf16=True,
            logging_steps=1,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=0,
            eval_strategy="steps" if use_val else "no",
            save_strategy="steps",
            eval_steps=eval_save_steps if use_val else None,
            save_steps=eval_save_steps,
            lr_scheduler_type=lr_scheduler,
            output_dir=phase_dir,
            save_total_limit=2,
            load_best_model_at_end=False,
            metric_for_best_model=None,
            ddp_find_unused_parameters=False if ddp else None,
            report_to="none",
            run_name=None,
        )

    os.makedirs(output_dir, exist_ok=True)

    # ── Phase A: proj_warmup — LLM + LoRA frozen, only input_proj + score train ──
    if proj_warmup_epochs > 0:
        print(f"\n{'='*60}")
        print(f"PROJ WARMUP: {proj_warmup_epochs} epoch(s), lr={proj_warmup_lr}")
        print(f"LLM + LoRA frozen — only input_proj + score head train")
        print('='*60)

        for p in model.llama_model.parameters():
            p.requires_grad = False

        device = next(model.llama_model.parameters()).device
        model.input_proj = model.input_proj.to(device)
        model.score = model.score.cpu().float()
        for p in model.input_proj.parameters():
            p.requires_grad = True
        for p in model.score.parameters():
            p.requires_grad = True

        total_pw = steps_per_epoch * proj_warmup_epochs
        pw_args = _make_training_args(proj_warmup_epochs, proj_warmup_lr, total_pw, "proj_warmup")
        pw_trainer = PeftTrainer(
            model=model,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            args=pw_args,
            data_collator=data_collator,
        )
        pw_trainer.train()
        del pw_trainer
        torch.cuda.empty_cache()
        print("Projection warmup complete.")

    # ── Phase B: lora_finetune — LoRA unfrozen, joint training ───────────────
    print(f"\n{'='*60}")
    print(f"LORA FINETUNE: {lora_finetune_epochs} epoch(s), lr={lora_finetune_lr}")
    print(f"LoRA unfrozen — joint training of input_proj + LLM adapters")
    print('='*60)

    for name, p in model.llama_model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True

    model.score = model.score.cpu().float()

    total_lf = steps_per_epoch * lora_finetune_epochs if max_steps == -1 else max_steps
    lf_args = _make_training_args(lora_finetune_epochs, lora_finetune_lr, total_lf, "lora_finetune")
    lf_trainer = PeftTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=lf_args,
        data_collator=data_collator,
    )
    lf_trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    del lf_trainer
    torch.cuda.empty_cache()

    # ── Evaluation ────────────────────────────────────────────────────────────
    model.eval()
    model.score = model.score.cpu().float()

    topk = [1, 5, 10, 20, 100]
    results = {k: np.zeros(len(topk)) for k in ['Precision', 'Recall', 'MRR', 'MAP', 'NDCG']}

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
                exclude_index, exclude_items = [], []
                for range_i, its in enumerate(all_pos):
                    exclude_index.extend([range_i] * len(its))
                    exclude_items.extend(its)
                ratings[exclude_index, exclude_items] = -(1 << 10)

            elif task_type == 'sequential':
                if len(testData[u]) == 0:
                    continue
                selected_items = [testData[u][1]] + dataset.allPos[u]
                groundTruth = [[0]]
                device = next(model.llama_model.parameters()).device
                seq = testData[u][0][-dataset.maxlen:]
                inputs = torch.LongTensor(seq).to(device).unsqueeze(0)
                inputs_mask = torch.ones(inputs.shape, device=device)
                _, ratings = model.predict(inputs, inputs_mask)
                ratings = ratings[:, selected_items]

            _, ratings_K = torch.topk(ratings, k=topk[-1])
            ratings_K = ratings_K.cpu().numpy()
            r = getLabel(groundTruth, ratings_K)
            for j, k in enumerate(topk):
                pre, rec = RecallPrecision_atK(groundTruth, r, k)
                results['Precision'][j] += pre
                results['Recall'][j] += rec
                results['MRR'][j] += MRR_atK(groundTruth, r, k)
                results['MAP'][j] += MAP_atK(groundTruth, r, k)
                results['NDCG'][j] += NDCG_atK(groundTruth, r, k)

    for key in results:
        results[key] /= float(len(users))

    print(f'\nEvaluation results:')
    for j, k in enumerate(topk):
        print(f'@{k}: Precision={results["Precision"][j]:.4f}  Recall={results["Recall"][j]:.4f}  '
              f'MRR={results["MRR"][j]:.4f}  MAP={results["MAP"][j]:.4f}  NDCG={results["NDCG"][j]:.4f}')

    # Save metrics as JSON for easy comparison across pairs
    metrics_path = os.path.join(output_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({k: v.tolist() for k, v in results.items()}, f, indent=2)
    print(f"Saved eval metrics to {metrics_path}")

    # ── Save final model ──────────────────────────────────────────────────────
    model.llama_model.save_pretrained(output_dir, save_embedding_layers=False)
    model_path = os.path.join(output_dir, "adapter.pth")
    proj_state = {"input_proj": model.input_proj.state_dict(),
                  "score": model.score.state_dict()}
    if task_type == 'general':
        proj_state["user_proj"] = model.user_proj.state_dict()
    torch.save(proj_state, model_path)
    print(f"Saved PEFT adapter to {output_dir}")
    print(f"Saved projection heads to {model_path}")

    plot_training_curves(output_dir)


def plot_training_curves(output_dir: str):
    import glob

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots.")
        return

    # Collect log_history from both stage subdirs and root output_dir
    all_train, all_eval, all_lr = [], [], []
    step_offset = 0
    for subdir in ["proj_warmup", "lora_finetune", ""]:
        state_path = os.path.join(output_dir, subdir, "trainer_state.json") if subdir \
                     else os.path.join(output_dir, "trainer_state.json")
        if not os.path.exists(state_path):
            continue
        with open(state_path) as f:
            state = json.load(f)
        for entry in state.get("log_history", []):
            step = entry["step"] + step_offset
            if "loss" in entry and "eval_loss" not in entry:
                all_train.append((step, entry["loss"]))
            if "eval_loss" in entry:
                all_eval.append((step, entry["eval_loss"]))
            if "learning_rate" in entry:
                all_lr.append((step, entry["learning_rate"]))
        # Offset so stage2 steps continue from stage1
        if state.get("log_history"):
            step_offset += state["log_history"][-1]["step"]

    if not all_train and not all_eval:
        print(f"No log history found in {output_dir}, skipping plots.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    if all_train:
        t_steps, t_loss = zip(*all_train)
        axes[0].plot(t_steps, t_loss, label="Train Loss", color="steelblue", alpha=0.8)
    if all_eval:
        e_steps, e_loss = zip(*all_eval)
        axes[0].plot(e_steps, e_loss, label="Val Loss", color="orange", marker="o", linewidth=2)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    if all_lr:
        lr_steps, lr_vals = zip(*all_lr)
        axes[1].plot(lr_steps, lr_vals, color="green")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Learning Rate")
        axes[1].set_title("Learning Rate Schedule")
        axes[1].grid(True)
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved training curves to {plot_path}")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    fire.Fire(train)
