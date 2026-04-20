"""
merge_embeddings.py
───────────────────
Merges embeddings.csv + embeddings_sub_*.csv for a given embedder node
into a single sorted embeddings_merged.csv.

Priority: embeddings.csv (original run) wins on duplicates — sub CSVs only
contribute tracks not already present in the original file.

Usage:
  python merge_embeddings.py                        # defaults to node_1
  python merge_embeddings.py --embedder_id 1
"""

import argparse
import os
import glob
import pandas as pd

_OUTPUT_ROOT = "/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings"

parser = argparse.ArgumentParser()
parser.add_argument("--embedder_id", type=int, default=1)
args = parser.parse_args()

node_dir    = os.path.join(_OUTPUT_ROOT, f"node_{args.embedder_id}")
main_csv    = os.path.join(node_dir, "embeddings.csv")
sub_pattern = os.path.join(node_dir, "embeddings_sub_*.csv")
out_csv     = os.path.join(node_dir, "embeddings_merged.csv")

# ── Load main CSV first (highest priority) ────────────────────────────────────
print(f"Reading main CSV: {main_csv}")
main_df = pd.read_csv(main_csv)
print(f"  {len(main_df):,} rows  |  track_ids {main_df['track_index'].min()} – {main_df['track_index'].max()}")

# ── Load sub CSVs ─────────────────────────────────────────────────────────────
sub_files = sorted(glob.glob(sub_pattern))
if not sub_files:
    print("No sub CSVs found. Nothing to merge.")
    raise SystemExit(0)

sub_dfs = []
for f in sub_files:
    df = pd.read_csv(f)
    print(f"Reading {os.path.basename(f)}: {len(df):,} rows  |  "
          f"track_ids {df['track_index'].min()} – {df['track_index'].max()}")
    sub_dfs.append(df)

sub_df = pd.concat(sub_dfs, ignore_index=True)

# ── Merge: original wins on duplicates ───────────────────────────────────────
# Keep only sub rows whose track_index is NOT already in the main CSV.
already_done = set(main_df["track_index"].tolist())
new_rows     = sub_df[~sub_df["track_index"].isin(already_done)]
print(f"\nSub CSVs total  : {len(sub_df):,} rows")
print(f"Already in main : {len(sub_df) - len(new_rows):,} duplicates dropped")
print(f"New rows added  : {len(new_rows):,}")

merged = (
    pd.concat([main_df, new_rows], ignore_index=True)
    .drop_duplicates(subset=["track_index"])
    .sort_values("track_index")
    .reset_index(drop=True)
)

print(f"\nFinal merged    : {len(merged):,} rows  |  "
      f"track_ids {merged['track_index'].min()} – {merged['track_index'].max()}")

merged.to_csv(out_csv, index=False)
print(f"\nSaved → {out_csv}")
