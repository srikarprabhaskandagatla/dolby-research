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

INSTRUMENTAL_FEATURES = [
    "Drum Set",
    "Drum Aggressiveness",
    "Synthetic Drums",
    "Percussion",
    "Electric Guitar",
    "Electric Guitar Distortion",
    "Acoustic Guitar",
    "String Ensemble",
    "Horn Ensemble",
    "Piano",
    "Organ",
    "Rhodes",
    "Synthesizer",
    "Synth Timbre",
    "Bass Guitar",
    "Reed Instrument",
]

FEATURE_DEFINITIONS = {
    "Drum Set": (
        "Presence and dominance of a drum set in the instrumentation. "
        "0.0=no drum set, 1.0=drum set is the dominant element."
    ),
    "Drum Aggressiveness": (
        "Aggressiveness of the drum set performance. "
        "0.0=extremely light (e.g. brushes, feather-light touch), "
        "1.0=extreme aggression common in hard rock and metal."
    ),
    "Synthetic Drums": (
        "Presence and dominance of synthetic drums (MIDI pads, drum machines, or programmed beats). "
        "0.0=no synthetic drums, 1.0=fully synthetic/programmed drums dominant. "
        "Note: Synthetic Drums is a subset of Drum Set — score 0.0 if Drum Set is absent."
    ),
    "Percussion": (
        "Presence and dominance of percussion instruments excluding drum set "
        "(e.g. congas, bongos, tambourine, shakers, marimba). "
        "0.0=no percussion, 1.0=percussion is the dominant element."
    ),
    "Electric Guitar": (
        "Presence and dominance of electric guitar(s) in the instrumentation. "
        "0.0=no electric guitar, 1.0=electric guitar is the dominant element."
    ),
    "Electric Guitar Distortion": (
        "Overall degree and impact of guitar distortion. "
        "0.0=very clean/unmodified tone, 1.0=extremely dirty tone common in extreme metal."
    ),
    "Acoustic Guitar": (
        "Presence and dominance of acoustic guitar(s) in the instrumentation. "
        "0.0=no acoustic guitar, 1.0=acoustic guitar is the dominant element."
    ),
    "String Ensemble": (
        "Presence and dominance of a string ensemble (from two violins to a full orchestra). "
        "0.0=no strings, 1.0=strings are the dominant element."
    ),
    "Horn Ensemble": (
        "Presence and dominance of a horn ensemble (from two trumpets to a full concert band). "
        "0.0=no horns, 1.0=horns are the dominant element."
    ),
    "Piano": (
        "Presence and dominance of piano in the instrumentation. "
        "0.0=no piano, 1.0=piano is the dominant element."
    ),
    "Organ": (
        "Presence and dominance of organ in the instrumentation. "
        "0.0=no organ, 1.0=organ is the dominant element."
    ),
    "Rhodes": (
        "Presence and dominance of a Fender Rhodes or other electric piano. "
        "0.0=no Rhodes/electric piano, 1.0=Rhodes is the dominant element."
    ),
    "Synthesizer": (
        "Presence and dominance of synthesizer(s), excluding synths mimicking other instruments "
        "(horns, flutes, electric pianos, strings, etc.). "
        "0.0=no synthesizer, 1.0=synthesizer is the dominant element."
    ),
    "Synth Timbre": (
        "Timbral character of synthesizers present in the track. "
        "0.0=ambient/atmospheric pads, 1.0=industrial/robotic timbres common in techno and electronic music. "
        "Score 0.0 if no synthesizer is present."
    ),
    "Bass Guitar": (
        "Presence and dominance of bass guitar in the instrumentation. "
        "0.0=no bass guitar, 1.0=bass guitar is the dominant element."
    ),
    "Reed Instrument": (
        "Presence and dominance of reed instruments (saxophone, clarinet, oboe, english horn, etc.). "
        "0.0=no reed instruments, 1.0=reed instruments are the dominant element."
    ),
}

SYSTEM_PROMPT = (
    "You are a music analyst. Output ONLY a valid JSON object with float scores 0.0–1.0. "
    "No explanation, no markdown, no extra text before or after the JSON."
)


def build_prompt(artist: str, title: str) -> list[dict]:
    keys_str = '", "'.join(INSTRUMENTAL_FEATURES)
    feature_block = "\n".join(
        f'- "{feat}": {FEATURE_DEFINITIONS[feat]}'
        for feat in INSTRUMENTAL_FEATURES
    )
    user_content = (
        f'Track: "{title}" by {artist}\n\n'
        f'Rate each instrumental feature 0.0–1.0 (continuous float) based on your knowledge of this track:\n'
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
MAX_NEW_TOKENS = 512


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
        for feat in INSTRUMENTAL_FEATURES:
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

csv_path = os.path.join(OUTPUT_DIR, f"instrument_features_part{args.part}.csv")

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
