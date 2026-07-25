"""
Convert the filtered LastFM-1K interactions CSV (produced by
data_preproc/data_preprocessing.py) into the sequential format that
E4SRec expects:

  datasets/sequential/LastFM/LastFM.txt
  datasets/sequential/LastFM/LastFM_sample.txt
  datasets/sequential/LastFM/item_id_master_map.csv

Run *before* SASRec baseline training and before zeroshot / finetune.
"""

import os
import random

import polars as pl

PREPROCESSED_CSV = (
    "/work/pi_dagarwal_umass_edu/project_7/hmagapu/"
    "user_sessions_lastfm1k_minuser1000_minitem7_sessgap1200_minsesslen10_minhist50.csv"
)

OUTPUT_DIR = "datasets/sequential/LastFM"
MIN_SEQ_LEN = 5
NUM_CANDIDATES = 200
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(RANDOM_SEED)

# ── 1. Load preprocessed CSV (no header) ────────────────────────────
print(f"Loading {PREPROCESSED_CSV} ...")
data = pl.read_csv(
    PREPROCESSED_CSV,
    has_header=False,
    new_columns=[
        "user_id", "ts", "art_id", "artist_name",
        "track_id", "track_name", "session_id",
    ],
)
print(f"  Loaded {data.height} rows")

# ── 2. Parse timestamps & sort chronologically ──────────────────────
data = (
    data
    .with_columns(
        pl.col("ts")
        .str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False)
        .alias("ts_dt")
    )
    .filter(pl.col("ts_dt").is_not_null())
    .sort(["user_id", "ts_dt"])
)

# ── 3. Build contiguous 1-indexed item mapping ──────────────────────
unique_tracks = (
    data.select(["track_id", "artist_name", "track_name"])
    .unique(subset=["track_id"])
    .sort("track_id")
)
item_map = {tid: idx + 1 for idx, tid in enumerate(unique_tracks["track_id"].to_list())}
total_items = len(item_map)
print(f"  Unique items: {total_items}")

# ── 4. Map items and build per-user chronological sequences ─────────
data = data.with_columns(
    pl.col("track_id").replace_strict(item_map).alias("item_idx")
)

user_sequences: dict[str, list[int]] = {}
for row in data.select(["user_id", "item_idx"]).iter_rows():
    uid, iidx = row
    user_sequences.setdefault(uid, []).append(iidx)

# ── 5. Filter short sequences & assign contiguous 1-indexed user IDs
final_sequences: dict[int, list[int]] = {}
u_count = 1
for uid in sorted(user_sequences):
    seq = user_sequences[uid]
    if len(seq) >= MIN_SEQ_LEN:
        final_sequences[u_count] = seq
        u_count += 1

total_users = u_count - 1
print(f"  Users with >= {MIN_SEQ_LEN} interactions: {total_users}")

# ── 6. Write item_id_master_map.csv ─────────────────────────────────
master_rows = []
for tid, artist, track in zip(
    unique_tracks["track_id"].to_list(),
    unique_tracks["artist_name"].to_list(),
    unique_tracks["track_name"].to_list(),
):
    master_rows.append({
        "item_id": item_map[tid],
        "track_id": tid,
        "artist_name": artist,
        "track_name": track,
    })
pl.DataFrame(master_rows).sort("item_id").write_csv(
    os.path.join(OUTPUT_DIR, "item_id_master_map.csv")
)

# ── 7. Write LastFM.txt ─────────────────────────────────────────────
print("Writing LastFM.txt ...")
with open(os.path.join(OUTPUT_DIR, "LastFM.txt"), "w") as f:
    for u_idx, items in final_sequences.items():
        f.write(f"{u_idx} " + " ".join(map(str, items)) + "\n")

# ── 8. Write LastFM_sample.txt (1 positive + 199 negatives) ────────
print("Writing LastFM_sample.txt ...")
with open(os.path.join(OUTPUT_DIR, "LastFM_sample.txt"), "w") as f:
    for u_idx, items in final_sequences.items():
        test_item = items[-1]
        user_items_set = set(items)
        candidates = [test_item]
        while len(candidates) < NUM_CANDIDATES:
            neg = random.randint(1, total_items)
            if neg not in user_items_set and neg not in candidates:
                candidates.append(neg)
        f.write(f"{u_idx} " + " ".join(map(str, candidates)) + "\n")

print("SUCCESS")
print(f"  User ID range: 1 → {total_users}")
print(f"  Item ID range: 1 → {total_items}")
print(f"  Output dir:    {OUTPUT_DIR}")
