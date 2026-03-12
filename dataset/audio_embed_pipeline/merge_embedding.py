import os, sys, torch
import pandas as pd

sys.path.append("/work/pi_dagarwal_umass_edu/project_7/srikar/dolby-research/dataset/extract_audio_pipeline/")
from paths import OUTPUT_MERGE_DIR

TOTAL_CHUNKS = 50
FINAL_DIR    = os.path.join(OUTPUT_MERGE_DIR, "final")
os.makedirs(FINAL_DIR, exist_ok=True)

PT_KEYS = [
    "audio_clap", "audio_mert", "audio_music2vec",
    "audio_encodec", "audio_mfcc",
    "text_minilm", "text_bgem3", "text_mpnet",
    "text_multilingual", "text_bert",
]


def merge_pt(key):
    # Merge all per-chunk .pt files for one embedder into a single file
    merged = {
        "track_ids":    [],
        "artist_names": [],
        "track_names":  [],
        "embeddings":   None,
    }

    loaded = 0
    for chunk_id in range(TOTAL_CHUNKS):
        path = os.path.join(OUTPUT_MERGE_DIR, f"chunk_{chunk_id:02d}", f"{key}.pt")
        if not os.path.exists(path):
            print(f"    [WARN] Missing: {path}")
            continue

        store = torch.load(path, weights_only=False)
        if not store["track_ids"]:
            continue

        merged["track_ids"]    += store["track_ids"]
        merged["artist_names"] += store["artist_names"]
        merged["track_names"]  += store["track_names"]

        emb = store["embeddings"]   # [N, dim]
        merged["embeddings"] = emb if merged["embeddings"] is None \
                               else torch.cat([merged["embeddings"], emb], dim=0)
        loaded += 1

    out_path = os.path.join(FINAL_DIR, f"{key}.pt")
    torch.save(merged, out_path)
    n = len(merged["track_ids"])
    shape = list(merged["embeddings"].shape) if merged["embeddings"] is not None else []
    print(f"  {key:22s} {shape}  ({n} tracks) are {loaded} chunks merged")
    return n


def merge_csv():
    # Merge all per-chunk master_lyrics.csv files into one
    dfs = []
    for chunk_id in range(TOTAL_CHUNKS):
        path = os.path.join(OUTPUT_MERGE_DIR, f"chunk_{chunk_id:02d}", "master_lyrics.csv")
        if not os.path.exists(path):
            print(f"    [WARN] Missing CSV: {path}")
            continue
        dfs.append(pd.read_csv(path))

    if not dfs:
        print("  [ERROR] No CSV chunks found.")
        return 0

    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Drop any duplicates in case of overlapping resumption
    merged_df = merged_df.drop_duplicates(subset=["track_index"])
    merged_df = merged_df.sort_values("track_index").reset_index(drop=True)

    out_path = os.path.join(FINAL_DIR, "master_lyrics.csv")
    merged_df.to_csv(out_path, index=False)
    print(f"  master_lyrics.csv     {len(merged_df)} tracks  → {out_path}")
    return len(merged_df)


if __name__ == "__main__":
    print(f"Merging {TOTAL_CHUNKS} chunks into final output")
    print(f"Output: {FINAL_DIR}")

    # Check which chunks completed
    completed = []
    missing   = []
    for chunk_id in range(TOTAL_CHUNKS):
        csv_path = os.path.join(OUTPUT_MERGE_DIR, f"chunk_{chunk_id:02d}", "master_lyrics.csv")
        if os.path.exists(csv_path):
            n = len(pd.read_csv(csv_path))
            completed.append((chunk_id, n))
        else:
            missing.append(chunk_id)

    print(f"Completed chunks : {len(completed)}/{TOTAL_CHUNKS}")
    if missing:
        print(f"Missing chunks   : {missing}")
        print(f"[WARN] Proceeding with available chunks only.\n")
    else:
        print(f"All chunks present.\n")

    # Merge .pt files
    print("Merging .pt files")
    for key in PT_KEYS:
        merge_pt(key)

    # Merge CSV
    print("\nMerging master_lyrics.csv")
    total = merge_csv()

    # Summary
    print(f"\n{'-'*64}")
    print(f"MERGE COMPLETE - {total} tracks total")
    print(f"Final outputs in: {FINAL_DIR}")