#!/usr/bin/env python3
import argparse
import json
import os
import re
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--part", type=int, default=0, choices=[0, 1],
                    help="0 = first 25K tracks, 1 = second 25K tracks")
args = parser.parse_args()

MODEL_PATH = "/work/pi_dagarwal_umass_edu/project_7/srikar/models/qwen2.5-7b-instruct"
INPUT_CSV  = "/work/pi_dagarwal_umass_edu/project_7/hmagapu/metadata/shared/top_50k_songs.csv"
OUTPUT_DIR = "/work/pi_dagarwal_umass_edu/project_7/srikar/dolby-research/metadata/metadata_output_v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

VOCAL_FEATURES = [
    "Vocal Register",
    "Vocal Timbre Thin to Full",
    "Vocal Breathiness",
    "Vocal Smoothness",
    "Vocal Grittiness",
    "Vocal Nasality",
    "Vocal Accompaniment",
]

FEATURE_DEFINITIONS = {
    "Vocal Register": (
        "The pitch range of the lead vocalist. "
        "0.0=very low/bass, 1.0=very high/soprano. "
        "Reference: Male midrange is A below middle C to E above middle C; "
        "female is approximately a fourth higher."
    ),
    "Vocal Timbre Thin to Full": (
        "The thickness/richness of vocal tone. "
        "0.0=thin/wispy/weak fundamental, 1.0=full/resonant/strong fundamental. "
        "Examples: 0.0=Olivia Newton-John, Margot Timmins; 1.0=Aretha Franklin, Josh Groban."
    ),
    "Vocal Breathiness": (
        "Presence of aspiration (breath) with lightness or airiness. "
        "0.0=not breathy/clean tone, 1.0=very breathy/airy. "
        "Examples of high breathiness: Hope Sandoval (Mazzy Star), Mariah Carey, Nick Drake."
    ),
    "Vocal Smoothness": (
        "Absence of roughness/raspiness, yielding roundness or sweetness. "
        "0.0=very rough/raspy, 1.0=very smooth/polished. "
        "Examples of high smoothness: Sade, Natalie Cole, James Taylor, Mel Torme."
    ),
    "Vocal Grittiness": (
        "Roughness/raspiness yielding harshness or dirtiness. "
        "0.0=completely clean, 1.0=extremely gritty/distorted. "
        "Examples of high grittiness: Tom Waits, Joe Cocker, Janis Joplin."
    ),
    "Vocal Nasality": (
        "Pinched or plugged-up quality, especially on nasal consonants (n, m, d). "
        "0.0=not nasal, 1.0=very nasal. "
        "Examples of high nasality: Bob Dylan, Joe Walsh, Reba McEntire."
    ),
    "Vocal Accompaniment": (
        "Level of dominance of accompaniment vocals — any vocal activity not in a lead role. "
        "0.0=no backing vocals, 1.0=very prominent/dominant backing vocals."
    ),
}

SYSTEM_PROMPT = (
    "You are a music analyst. Output ONLY a valid JSON object with float scores 0.0–1.0. "
    "No explanation, no markdown, no extra text before or after the JSON."
)


def build_prompt(artist: str, title: str) -> list[dict]:
    keys_str = '", "'.join(VOCAL_FEATURES)
    feature_block = "\n".join(
        f'- "{feat}": {FEATURE_DEFINITIONS[feat]}'
        for feat in VOCAL_FEATURES
    )
    user_content = (
        f'Track: "{title}" by {artist}\n\n'
        f'Rate each vocal feature 0.0–1.0 (continuous float) based on your knowledge of this track:\n'
        f'{feature_block}\n\n'
        f'Return a JSON object with exactly these keys: "{keys_str}"'
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


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

BATCH_SIZE = 8
MAX_NEW_TOKENS = 256


def annotate_batch(rows: list[dict]) -> list[dict | None]:
    prompts = [build_prompt(r["artist_name"], r["track_name"]) for r in rows]
    texts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
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
        for feat in VOCAL_FEATURES:
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


CHECKPOINT_EVERY = 10_000

print("Loading full input data")
full_df = pd.read_csv(INPUT_CSV)

half = len(full_df) // 2
if args.part == 0:
    part_df = full_df.iloc[:half]
else:
    part_df = full_df.iloc[half:]

csv_path = os.path.join(OUTPUT_DIR, f"vocal_features_part{args.part}.csv")

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
