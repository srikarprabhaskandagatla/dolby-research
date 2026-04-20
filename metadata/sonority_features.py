#!/usr/bin/env python3
import argparse
import json
import os
import re
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

sonority_features_df = pd.read_csv(GENES_TSV, sep="\t").iloc[43:49]
SONORITY_FEATURES = list(sonority_features_df["name"])

SONORITY_FEATURE_INDICES = {
    "Live Recording":    43,
    "Audio Production":  44,
    "Aural Intensity":   45,
    "Acoustic Sonority": 46,
    "Electric Sonority": 47,
    "Synthetic Sonority":48,
}

FEATURE_DEFINITIONS = {
    "Live Recording": (
        "Indicates whether the song was recorded in a studio or at a live performance. "
        "0.0=definitively studio-recorded (no live cues whatsoever). "
        "0.4–0.6=ambiguous — title or artist suggests possible live element (e.g. 'Unplugged', 'Acoustic Session', 'In Concert', 'BBC Session'). "
        "0.8–1.0=clearly live — title contains 'Live', 'Live at [venue]', 'Live [year/tour]', or is a parenthetical medley cover of another artist's song. "
        "Base your score purely on title and known release context, not artist reputation."
    ),
    "Audio Production": (
        "Evaluates the quality and sophistication of the audio production on a continuous scale. "
        "A middling score (~0.5) indicates a typical commercial pop production — clean and well-executed but unexceptional. "
        "0.0–0.2=very poor fidelity: heavy hiss, pops, overpowering noise, archival or unlistenable quality. "
        "0.2–0.4=below standard: lo-fi, demo-quality, or early mono-era recordings (pre-1964). "
        "0.5=standard commercial default for 1964–present properly released recordings. "
        "0.6–0.8=production exceeds standard — distinguished craft, notable producer, audiophile quality (e.g. Beatles post-1966, Pink Floyd, iconic soul). "
        "0.9–1.0=landmark production masterpiece, widely considered a pinnacle of recorded sound."
    ),
    "Aural Intensity": (
        "Measures the overall softness or loudness and sonic aggression of the song on a continuous scale. "
        "0.0=extremely quiet, near-silent, or completely unobtrusive (ambient, silence). "
        "0.2=very sparse and delicate (solo acoustic, slow meditative jazz). "
        "0.4=gentle and restrained (folk ballad, slow R&B, quiet singer-songwriter). "
        "0.5=standard intensity — default for most released pop, rock, R&B, indie, blues. "
        "0.7=energetic (uptempo rock, driving punk, danceable pop, fast rhythm). "
        "0.9–1.0=extreme loudness and aggression (thrash metal, grindcore, noise rock)."
    ),
    "Acoustic Sonority": (
        "Indicates the presence and degree of dominance of acoustic sonorities — acoustic piano, acoustic guitar, human voice, orchestral instruments. "
        "0.0=purely electronic or synthetic; no acoustic instruments present at all. "
        "0.2=trace acoustic element in an otherwise electric/electronic song. "
        "0.4=acoustic instruments present but secondary to electric/electronic sounds. "
        "0.6=balanced mix of acoustic and electric/electronic (folk-rock, acoustic-electric). "
        "0.8=predominantly acoustic — most instruments are acoustic (singer-songwriter, unplugged). "
        "1.0=purely acoustic — classical, solo guitar/piano, traditional folk, a cappella; no electric or synthetic sounds."
    ),
    "Electric Sonority": (
        "Indicates the presence and degree of dominance of electric instruments — electric guitar, electric bass, electric organ, electric piano, amplified instruments. "
        "0.0=no electric instruments at all (purely acoustic or purely synthetic). "
        "0.2=trace electric presence in an otherwise acoustic or synthetic song. "
        "0.4=electric instruments audible but secondary. "
        "0.6=electric instruments are significant and central (mainstream rock, blues, funk). "
        "0.8=dominant electric sound (classic rock, hard rock, heavy blues). "
        "1.0=overwhelming — electric guitar or electric instruments are the entire sonic identity (guitar-driven heavy rock)."
    ),
    "Synthetic Sonority": (
        "Indicates the presence and degree of dominance of synthesizers, drum machines, and electronic/synthetic sounds. "
        "0.0=no synthetic sounds whatsoever — purely organic instruments (rock, jazz, classical, folk). "
        "0.2=subtle synth pad or electronic texture buried in an organic mix. "
        "0.4=audible synthesizer or electronic elements in a supporting role. "
        "0.6=significant electronic production (synth-pop, electro-R&B, modern pop with heavy synths). "
        "0.8=dominant electronic/synthetic texture (EDM-pop, heavy electronic production). "
        "1.0=fully synthetic — techno, industrial, pure electronic music with no organic instruments."
    ),
}

SYSTEM_PROMPT = (
    "You are a music analyst. Output ONLY a valid JSON object with float scores 0.0–1.0. "
    "No explanation, no markdown, no extra text before or after the JSON."
)


def build_prompt(artist: str, title: str) -> list[dict]:
    keys_str = '", "'.join(SONORITY_FEATURES)
    user_content = (
        f'Track: "{title}" by {artist}\n\n'
        f'Rate each feature 0.0–1.0 (continuous float) based on your knowledge of this track\'s genre, era, and style:\n'
        f'- "Live Recording": 0.0=studio recording, 1.0=live performance\n'
        f'- "Audio Production": 0.0=very poor fidelity, 0.5=standard commercial, 1.0=landmark production\n'
        f'- "Aural Intensity": 0.0=silent/ambient, 0.5=standard pop/rock, 1.0=extreme loud/aggressive\n'
        f'- "Acoustic Sonority": 0.0=no acoustic instruments, 1.0=purely acoustic\n'
        f'- "Electric Sonority": 0.0=no electric instruments, 1.0=overwhelmingly electric\n'
        f'- "Synthetic Sonority": 0.0=no synths/electronics, 1.0=fully synthetic\n\n'
        f'Return a JSON object with exactly these keys: "{keys_str}"'
    )
    return [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_content},
    ]

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_json(text: str) -> dict | None:
    text = text.strip()
    # model may have continued from prefill '{' — prepend if missing
    if text and not text.startswith("{"):
        text = "{" + text
    for candidate in [text, _JSON_RE.search(text) and _JSON_RE.search(text).group()]:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # try closing an unclosed object (truncated output)
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
        for feat in SONORITY_FEATURES:
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

# print("Loading validation data")
# validation_df = pd.read_csv(VALIDATION_CSV)
# validation_df["gene_values"] = validation_df["gene_values"].apply(
#     lambda x: ast.literal_eval(x) if isinstance(x, str) else x
# )

# N_TEST   = 100         
# test_subset = validation_df.head(N_TEST)
# all_rows = test_subset.to_dict("records")

CHECKPOINT_EVERY = 10_000

print("Loading full input data")
full_df = pd.read_csv(INPUT_CSV)

# Split into two halves based on --part
half = len(full_df) // 2
if args.part == 0:
    part_df = full_df.iloc[:half]
else:
    part_df = full_df.iloc[half:]

csv_path = os.path.join(OUTPUT_DIR, f"sonority_features_part{args.part}.csv")

# Resume: skip already-annotated tracks
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
total_done = len(done_keys)

for batch_start in range(0, len(all_rows), BATCH_SIZE):
    batch = all_rows[batch_start : batch_start + BATCH_SIZE]
    batch_results = annotate_batch(batch)

    for row, scores in zip(batch, batch_results):
        if scores is not None:
            entry = {"artist_name": row["artist_name"], "track_name": row["track_name"]}
            entry.update(scores)
            checkpoint_buf.append(entry)

    total_done += len(batch)
    completed = min(batch_start + BATCH_SIZE, len(all_rows))
    print(f"  [{completed}/{len(all_rows)}] completed")

    if len(checkpoint_buf) >= CHECKPOINT_EVERY:
        write_header = not os.path.exists(csv_path)
        pd.DataFrame(checkpoint_buf).to_csv(csv_path, mode="a", index=False, header=write_header)
        print(f"  ✓ Checkpoint saved ({len(checkpoint_buf)} rows) → {csv_path}")
        checkpoint_buf = []

# Final flush
if checkpoint_buf:
    write_header = not os.path.exists(csv_path)
    pd.DataFrame(checkpoint_buf).to_csv(csv_path, mode="a", index=False, header=write_header)
    print(f"  ✓ Final checkpoint saved ({len(checkpoint_buf)} rows) → {csv_path}")

print(f"\n✓ Part {args.part} done. Output: {csv_path}")