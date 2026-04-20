import pandas as pd
import os

INPUT_DIR = "metadata_output_v2"
OUTPUT_DIR = "metadata_output_v2"

FEATURE_TYPES = ["harmony_features", "lyrics_features", "rhythm_features", "sonority_features"]


def merge():
    for feature in FEATURE_TYPES:
        part0 = pd.read_csv(os.path.join(INPUT_DIR, f"{feature}_part0.csv"))
        part1 = pd.read_csv(os.path.join(INPUT_DIR, f"{feature}_part1.csv"))

        merged = pd.concat([part0, part1], ignore_index=True)
        merged.insert(0, "track_index", range(len(merged)))

        out_path = os.path.join(OUTPUT_DIR, f"{feature}_merged.csv")
        merged.to_csv(out_path, index=False)
        print(f"Saved {out_path} ({len(merged)} rows)")


if __name__ == "__main__":
    merge()
