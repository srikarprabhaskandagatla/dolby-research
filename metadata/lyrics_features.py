import argparse
import json
import re
import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--part", type=int, default=0, choices=[0, 1],
                    help="0 = first 25K tracks, 1 = second 25K tracks")
args = parser.parse_args()

MODEL_PATH = "/work/pi_dagarwal_umass_edu/project_7/srikar/models/qwen2.5-7b-instruct"
GENES_TSV  = "/work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/mgphot_genes.tsv"
INPUT_CSV  = "/work/pi_dagarwal_umass_edu/project_7/hmagapu/metadata/shared/top_50k_songs.csv"
OUTPUT_DIR = "/work/pi_dagarwal_umass_edu/project_7/srikar/dolby-research/metadata/metadata_output_v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print(f"Model loaded on: {next(model.parameters()).device}\n")

BATCH_SIZE = 16
MAX_NEW_TOKENS = 256

# ── Feature definitions ────────────────────────────────────────────────────────

lyrics_features = pd.read_csv(GENES_TSV, sep="\t").iloc[35:43]
LYRICS_FEATURES = []
for idx, row in lyrics_features.iterrows():
    print(row['name'])
    print(row['description'])
    print("**********************")
    LYRICS_FEATURES.append(row['name'])

FEATURE_DEFINITIONS = {
    'Angry Lyrics': (
        'Presence and dominance of angry lyrics in the song. '
        '0 = no anger; 1 = mild irritation or frustration (e.g. breakup annoyance); '
        '2 = open confrontation or resentment; 3 = anger is a central, recurring theme; '
        '4 = intense rage or aggression dominant; 5 = anger pervades every line (e.g. hardcore, metal rants). '
        'Score conservatively — most pop/rock scores 0–2.'
    ),
    'Sad Lyrics': (
        'The presence and degree of dominance of sad lyrics in the song. '
        'Sad lyrics express melancholy, sorrow, grief, despair, or unhappiness. '
        '0 = no sad lyrics, 5 = sad lyrics are dominant throughout the song.'
    ),
    'Happy/Joyful Lyrics': (
        'The presence and degree of dominance of happy or joyful lyrics in the song. '
        'Happy/joyful lyrics express joy, happiness, celebration, contentment, or positivity. '
        '0 = no happy/joyful lyrics, 5 = happy/joyful lyrics are dominant throughout the song.'
    ),
    'Humorous Lyrics': (
        'The presence and degree of dominance of funny or humorous lyrics in the song. '
        'Humorous lyrics use wit, comedy, satire, or sarcasm for entertainment or commentary. '
        '0 = no humorous lyrics, 5 = humorous lyrics are dominant throughout the song.'
    ),
    'Love/Romance Lyrics': (
        'The presence and degree of dominance of romantic lyrics or lyrics about love in the song. '
        'Love/romance lyrics express romantic feelings, attraction, devotion, or relationships. '
        '0 = no love/romance lyrics, 5 = love/romance lyrics are dominant throughout the song.'
    ),
    'Social/Political Lyrics': (
        'Presence of lyrics addressing social justice, politics, inequality, or societal commentary. '
        '0 = purely personal (love, lifestyle, introspection — no society); '
        '1 = passing reference to community, race, or world; '
        '2 = social theme is secondary; 3 = clear social/political message; '
        '4 = primarily a political statement; 5 = explicit protest anthem. '
        'Songs referencing life struggles, systemic issues, or community score at least 1–2.'
    ),
    'Abstract Lyrics': (
        'Presence of surreal, symbolic, or deliberately obscure imagery over literal meaning. '
        '0 = fully literal and direct; 1 = occasional metaphor; 2 = notably poetic or symbolic; '
        '3 = abstract imagery is the dominant mode; 4 = surreal or stream-of-consciousness; '
        '5 = entirely non-literal, dreamlike, or nonsensical. '
        'Most mainstream pop/rock scores 0–1. Score 2+ only when imagery is genuinely unusual or non-literal.'
    ),
    'Explicit Lyrics': (
        'A measure of the explicitness of lyrics based on listener, advertiser, and brand safety concerns. '
        'Evaluation is based on an established rubric assessing the presence of explicit words, '
        'as well as thematic material including descriptions of sex, violence, and other mature subject matter. '
        '0 = no explicit content, 5 = highly explicit throughout the song.'
    ),
}

LYRICS_FEATURE_INDICES = {
    'Angry Lyrics': 35,
    'Sad Lyrics': 36,
    'Happy/Joyful Lyrics': 37,
    'Humorous Lyrics': 38,
    'Love/Romance Lyrics': 39,
    'Social/Political Lyrics': 40,
    'Abstract Lyrics': 41,
    'Explicit Lyrics': 42,
}

SYSTEM_PROMPT = (
    "You are an expert music analyst annotating lyrical characteristics of songs. "
    "Rate each feature on a 0–5 integer scale using the provided definitions. "
    "Respond ONLY with a valid JSON object mapping feature names to integer scores. "
    "No explanation, no markdown, no extra keys."
)

# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_user_prompt(artist: str, title: str) -> str:
    feature_block = "\n".join(
        f'- "{feat}" (0–5): {FEATURE_DEFINITIONS[feat]}'
        for feat in LYRICS_FEATURES
    )
    return (
        f'Track: "{title}" by {artist}\n\n'
        f'Rate each lyrical feature from 0 (lowest) to 5 (highest):\n'
        f'{feature_block}\n\n'
        f'Return a JSON object with exactly these keys: {LYRICS_FEATURES}'
    )

# ── Annotation ─────────────────────────────────────────────────────────────────

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_json(text: str) -> dict | None:
    text = text.strip()
    if text and not text.startswith("{"):
        text = "{" + text
    for candidate in [text, _JSON_RE.search(text) and _JSON_RE.search(text).group()]:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return json.loads(candidate.rstrip(",") + "}")
            except json.JSONDecodeError:
                pass
    return None


def annotate_batch(rows: list[dict]) -> list[dict | None]:
    prompts = [
        [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": build_user_prompt(r["artist_name"], r["track_name"])},
        ]
        for r in rows
    ]

    texts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]

    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **encodings,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = encodings["input_ids"].shape[1]
    results = []
    for i, ids in enumerate(output_ids):
        raw_text = tokenizer.decode(ids[prompt_len:], skip_special_tokens=True)
        scores = extract_json(raw_text)

        if scores is None:
            print(f"JSON parse failed for: {rows[i]['track_name']} — raw: {raw_text[:120]}")
            results.append(None)
            continue

        normalized = {}
        ok = True
        for feat in LYRICS_FEATURES:
            val = scores.get(feat)
            if val is None:
                print(f"Missing key '{feat}' for: {rows[i]['track_name']}")
                ok = False
                break
            try:
                normalized[feat] = float(val) / 5.0
            except (TypeError, ValueError):
                print(f"Non-numeric value '{val}' for feature '{feat}'")
                ok = False
                break

        results.append(normalized if ok else None)

    return results

# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate lyrics annotations against ground truth gene values.
    
    Args:
        df: DataFrame with gene_values column and lyrics feature columns
    
    Returns:
        DataFrame with evaluation metrics (MAE, Spearman correlation)
    """
    rows = []
    for feat, idx in LYRICS_FEATURE_INDICES.items():
        gt = df["gene_values"].apply(lambda x: x[idx])
        pred = df[feat]
        mask = pred.notna()
        gt, pred = gt[mask], pred[mask]
        if len(gt) < 10:
            continue
        mae = (gt - pred).abs().mean()
        rho, _ = spearmanr(gt, pred)
        rows.append({
            "feature": feat,
            "n": len(gt),
            "MAE": round(mae, 4),
            "Spearman_rho": round(rho, 4),
        })
    return pd.DataFrame(rows).sort_values("Spearman_rho", ascending=False)


def check_bias(df: pd.DataFrame):
    """
    Check bias between ground truth and predicted values.
    
    Args:
        df: DataFrame with gene_values and lyrics feature columns
    """
    print(f"{'Feature':<30} {'GT mean':>8} {'Pred mean':>10} {'Bias':>8}")
    print("-" * 60)
    for feat, idx in LYRICS_FEATURE_INDICES.items():
        gt = df["gene_values"].apply(lambda x: x[idx])
        pred = df[feat].dropna()
        if len(pred) > 0:
            bias = pred.mean() - gt.mean()
            print(f"{feat:<30} {gt.mean():>8.3f} {pred.mean():>10.3f} {bias:>8.3f}")


# ── Main execution ─────────────────────────────────────────────────────────────

CHECKPOINT_EVERY = 10_000

print("Loading full input data")
full_df = pd.read_csv(INPUT_CSV)

half = len(full_df) // 2
part_df = full_df.iloc[:half] if args.part == 0 else full_df.iloc[half:]

csv_path = os.path.join(OUTPUT_DIR, f"lyrics_features_part{args.part}.csv")

if os.path.exists(csv_path):
    done_df = pd.read_csv(csv_path)
    done_keys = set(zip(done_df["artist_name"], done_df["track_name"]))
    print(f"Resuming: {len(done_keys)} tracks already done, skipping them.")
else:
    done_keys = set()

all_rows = [r for r in part_df.to_dict("records")
            if (r["artist_name"], r["track_name"]) not in done_keys]

print(f"Part {args.part}: {len(all_rows)} tracks remaining\n")

checkpoint_buf = []

for batch_start in range(0, len(all_rows), BATCH_SIZE):
    batch = all_rows[batch_start : batch_start + BATCH_SIZE]
    batch_results = annotate_batch(batch)

    for row, scores in zip(batch, batch_results):
        if scores is not None:
            entry = {"artist_name": row["artist_name"], "track_name": row["track_name"]}
            entry.update(scores)
            checkpoint_buf.append(entry)

    completed = min(batch_start + BATCH_SIZE, len(all_rows))
    print(f"  [{completed}/{len(all_rows)}] completed")

    if len(checkpoint_buf) >= CHECKPOINT_EVERY:
        write_header = not os.path.exists(csv_path)
        pd.DataFrame(checkpoint_buf).to_csv(csv_path, mode="a", index=False, header=write_header)
        print(f"  ✓ Checkpoint saved ({len(checkpoint_buf)} rows) → {csv_path}")
        checkpoint_buf = []

if checkpoint_buf:
    write_header = not os.path.exists(csv_path)
    pd.DataFrame(checkpoint_buf).to_csv(csv_path, mode="a", index=False, header=write_header)
    print(f"  ✓ Final checkpoint saved ({len(checkpoint_buf)} rows) → {csv_path}")

print(f"\n✓ Part {args.part} done. Output: {csv_path}")
