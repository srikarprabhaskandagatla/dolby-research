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

rhythm_features = pd.read_csv(GENES_TSV, sep="\t").iloc[9:19]
RHYTHM_FEATURES = []
for idx, row in rhythm_features.iterrows():
    print(row['name'])
    print(row['description'])
    print("**********************")
    RHYTHM_FEATURES.append(row['name'])

FEATURE_DEFINITIONS = {
    'Tempo': (
        'The basic tempo (beats per minute) of the song. Lower scores indicate slower tempos, '
        'higher scores indicate faster tempos. Note that factors beyond BPM affect perceived tempo, '
        'such as sparse drum subdivisions or a "cut time" feel which can make a song feel slower than its actual BPM.'
    ),
    'Cut Time Feel': (
        'A "two" feel where 4/4 measures are heard as 2 large beats, typically because the bass '
        'plays only on beats 1 and 3. Most pop/rock scores 0. '
        'Calibration: typical 4/4 rock/pop = 0; occasional half-time section = 1; '
        'strong half-time bass pattern throughout = 2–3; cut-time completely dominates = 4–5. '
        '0 = normal 4/4 feel, 5 = cut-time feel completely dominates.'
    ),
    'Triple Meter': (
        'Three-beat groupings such as 3/4 (waltz). Absent from almost all pop, rock, and electronic. '
        '0 = duple (2/4, 4/4) throughout, 5 = waltz or triple meter throughout.'
    ),
    'Compound Meter': (
        '6/8 or 12/8 time: beats group in 2 or 4 but each beat subdivides into 3. '
        'Distinct from triplet feel overlaid on 4/4. Rare in pop/rock. '
        '0 = absent; 3 = clearly compound but mixed with other meters; 5 = compound meter completely dominates.'
    ),
    'Odd Meter': (
        'Unusual beat groupings such as 5/4, 7/4, or 7/8. Found in prog-rock, jazz, world music. '
        'Extremely rare — score 0 for nearly all pop, rock, country, R&B, and electronic music. '
        '0 = absent (the vast majority of songs), 5 = odd meter pervasive throughout.'
    ),
    'Swing Feel': (
        'Uneven 8th notes where the first is longer than the second — the hallmark of jazz. '
        'Distinct from shuffle (which is more insistent). '
        'Absent from straight-8th genres (rock, electronic, most pop). '
        '0 = straight 8ths, 5 = heavy swing throughout.'
    ),
    'Shuffle Feel': (
        'Uneven 8th notes articulated insistently on every beat — blues/country groove. '
        'More driving and insistent than jazz swing. '
        'Calibration: pop/rock/electronic = 0; rock with slight shuffle influence = 1; '
        'blues-rock or country with clear shuffle = 2–3; pure blues/country shuffle dominates = 4–5. '
        '0 = absent (most songs), 5 = shuffle defines the entire groove.'
    ),
    'Syncopation Low to High': (
        'Rhythmic tension against the strong beats via anticipations or cross-rhythms, most salient in drums. '
        'Calibration: classical/straight rock ≈ 1–2; funk/R&B ≈ 3–4; heavily syncopated hip-hop/fusion ≈ 4–5. '
        '0 = every instrument locks to the beat, 5 = meter constantly obscured by cross-rhythms.'
    ),
    'Backbeat': (
        'Snare or clap accent on beats 2 and 4. '
        'IMPORTANT: Most rock/pop with a standard drum kit scores 2–3. '
        'Calibration: no drums/orchestral/classical = 0; '
        'jazz with soft brush 2&4 = 1–2; '
        'standard rock/pop drum kit = 2–3; '
        'pop/rock where the snare crack is loud and prominent = 3; '
        'soul/gospel/funk where backbeat is a defining feature = 3–4; '
        'extreme clap-or-snare backbeat completely dominates = 5. '
        '0 = no 2&4 accent, 5 = backbeat is the dominant rhythmic feature.'
    ),
    'Danceability': (
        'The relative "danceability" of the song. Danceability indicates how suitable a song is for dancing. '
        '0 = difficult or impossible to dance to, '
        '5 = highly danceable, specifically made for dancing.'
    ),
}

RHYTHM_FEATURE_INDICES = {
    'Tempo': 9,
    'Cut Time Feel': 10,
    'Triple Meter': 11,
    'Compound Meter': 12,
    'Odd Meter': 13,
    'Swing Feel': 14,
    'Shuffle Feel': 15,
    'Syncopation Low to High': 16,
    'Backbeat': 17,
    'Danceability': 18,
}

SYSTEM_PROMPT = (
    "You are an expert musicologist annotating rhythmic characteristics of songs. "
    "Rate each feature on a 0–5 integer scale using the provided definitions and examples. "
    "Respond ONLY with a valid JSON object mapping feature names to integer scores. "
    "No explanation, no markdown, no extra keys."
)

# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_user_prompt(artist: str, title: str) -> str:
    feature_block = "\n".join(
        f'- "{feat}" (0–5): {FEATURE_DEFINITIONS[feat]}'
        for feat in RHYTHM_FEATURES
    )
    return (
        f'Track: "{title}" by {artist}\n\n'
        f'Rate each rhythm feature from 0 (lowest) to 5 (highest):\n'
        f'{feature_block}\n\n'
        f'Return a JSON object with exactly these keys: {RHYTHM_FEATURES}'
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
        for feat in RHYTHM_FEATURES:
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
    
    rows = []
    for feat, idx in RHYTHM_FEATURE_INDICES.items():
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
        df: DataFrame with gene_values and rhythm feature columns
    """
    print(f"{'Feature':<30} {'GT mean':>8} {'Pred mean':>10} {'Bias':>8}")
    print("-" * 60)
    for feat, idx in RHYTHM_FEATURE_INDICES.items():
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

csv_path = os.path.join(OUTPUT_DIR, f"rhythm_features_part{args.part}.csv")

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
