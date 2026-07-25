import io
import os
import json
import pickle
import sys
from typing import List

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import fire
import numpy as np
import torch
from utils.prompter import Prompter
from model import LLM4Rec
from utils.data_utils import SequentialDataset
from utils.eval_utils import RecallPrecision_atK, MRR_atK, MAP_atK, NDCG_atK, getLabel


def evaluate(
    base_model: str = "/work/pi_dagarwal_umass_edu/project_7/srikar/dolby-research/models/qwen2.5-7b-instruct",
    data_path: str = "/work/pi_dagarwal_umass_edu/project_7/srikar/E4SRec/datasets/sequential/LastFM/",
    embed_pkl: str = "",  # path to multimodal embed pkl; empty = SASRec_item_embed.pkl
    cache_dir: str = "",
    cutoff_len: int = 128,
    load_in_4bit: bool = True,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    prompt_template_name: str = "alpaca",
):
    """Zero-shot evaluation: Qwen 2.5 7B + embeddings, no finetuned weights loaded."""

    print(f"base_model:  {base_model}")
    print(f"data_path:   {data_path}")
    print(f"embed_pkl:   {embed_pkl or '(default SASRec_item_embed.pkl)'}")
    print(f"load_in_4bit:{load_in_4bit}")

    prompter = Prompter(prompt_template_name)

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset = SequentialDataset(data_path, 50)

    # ── Load embeddings ───────────────────────────────────────────────────────
    embed_path = embed_pkl if embed_pkl else os.path.join(data_path, "SASRec_item_embed.pkl")
    if not os.path.exists(embed_path):
        raise FileNotFoundError(f"Embedding file not found: {embed_path}")

    class _CpuUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
            return super().find_class(module, name)

    with open(embed_path, "rb") as f:
        raw_embed = _CpuUnpickler(f).load().half()  # fp16 to halve CPU RAM

    input_dim = raw_embed.shape[1]
    raw_shape = raw_embed.shape
    n_items = len(dataset.item_map) + 1
    item_embed = torch.zeros(n_items, input_dim, dtype=torch.float16)
    for raw_id, new_idx in dataset.item_map.items():
        if raw_id < raw_embed.shape[0]:
            item_embed[new_idx] = raw_embed[raw_id]
    del raw_embed  # free before loading the 7B model

    print(f"Embedding shape: {raw_shape} → padded to [{n_items}, {input_dim}]")

    # ── Build model (zero-shot: pre-trained Qwen, random proj/score) ──────────
    model = LLM4Rec(
        base_model=base_model,
        task_type="sequential",
        cache_dir=cache_dir,
        input_dim=input_dim,
        output_dim=dataset.m_item,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        device_map="auto",
        load_in_4bit=load_in_4bit,
        instruction_text=prompter.generate_prompt("sequential"),
        user_embeds=None,
        input_embeds=item_embed.float(),
        max_seq_len=cutoff_len,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("Model init complete. Starting eval...", flush=True)
    model.eval()
    model.score = model.score.cpu().float()

    topk = [1, 5, 10, 20, 100]
    results = {k: np.zeros(len(topk)) for k in ["Precision", "Recall", "MRR", "MAP", "NDCG"]}

    testData = dataset.testData
    users = np.arange(dataset.n_user)
    n_evaluated = 0

    print(f"Starting evaluation over {len(users)} users...")
    with torch.inference_mode():
        for u in users:
            if len(testData.get(u, [])) == 0:
                continue
            if n_evaluated % 50 == 0:
                print(f"  Evaluated {n_evaluated} users...", flush=True)

            selected_items = [testData[u][1]] + dataset.allPos[u]
            groundTruth = [[0]]

            device = next(model.llama_model.parameters()).device
            seq = testData[u][0][-dataset.maxlen:]
            inputs = torch.LongTensor(seq).to(device).unsqueeze(0)
            inputs_mask = torch.ones(inputs.shape, device=device)

            # Score only selected_items instead of the full 372k catalogue.
            # This avoids a massive CPU matmul on every user.
            _, pooled = model._encode(inputs, inputs_mask)
            pooled_cpu = pooled.detach().cpu().float()
            W = model.score.weight[selected_items]  # [n_candidates, hidden]
            ratings = (W @ pooled_cpu.squeeze()).unsqueeze(0)  # [1, n_candidates]

            _, ratings_K = torch.topk(ratings, k=min(topk[-1], len(selected_items)))
            ratings_K = ratings_K.cpu().numpy()
            r = getLabel(groundTruth, ratings_K)

            for j, k in enumerate(topk):
                pre, rec = RecallPrecision_atK(groundTruth, r, k)
                results["Precision"][j] += pre
                results["Recall"][j] += rec
                results["MRR"][j] += MRR_atK(groundTruth, r, k)
                results["MAP"][j] += MAP_atK(groundTruth, r, k)
                results["NDCG"][j] += NDCG_atK(groundTruth, r, k)
            n_evaluated += 1

    for key in results:
        results[key] /= float(max(n_evaluated, 1))

    print(f"\nZero-shot evaluation ({n_evaluated} users):")
    for j, k in enumerate(topk):
        print(f"  @{k}: Precision={results['Precision'][j]:.4f}  Recall={results['Recall'][j]:.4f}  "
              f"MRR={results['MRR'][j]:.4f}  MAP={results['MAP'][j]:.4f}  NDCG={results['NDCG'][j]:.4f}")

    # Save results alongside the embed pkl or in cwd
    out_dir = os.path.dirname(embed_pkl) if embed_pkl else "."
    metrics_path = os.path.join(out_dir, "zeroshot_eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({k: v.tolist() for k, v in results.items()}, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    fire.Fire(evaluate)
