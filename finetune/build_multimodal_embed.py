import argparse
import io
import os
import pickle

import pandas as pd
import torch

BATCH1_ROOT = "/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1"
DATA_PATH   = "/work/pi_dagarwal_umass_edu/project_7/srikar/E4SRec/datasets/sequential/LastFM"
TOP_K       = 50_000
SASREC_PKL  = os.path.join(DATA_PATH, "SASRec_item_embed.pkl")

AUDIO_NODES  = {0: "node_0", 1: "node_1", 2: "node_2", 3: "node_3", 4: "node_4"}
LYRICS_NODES = {0: "node_5", 1: "node_6", 2: "node_7", 3: "node_8", 4: "node_9"}

AUDIO_NAMES  = {0: "clap", 1: "mert", 2: "music2vec", 3: "encodec", 4: "mfcc"}
LYRICS_NAMES = {0: "minilm", 1: "bgem3", 2: "mpnet", 3: "multilingual", 4: "bert"}


class _CpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def load_sasrec(path: str) -> torch.Tensor:
    print(f"  Loading SASRec embeddings from {path} ...")
    with open(path, "rb") as f:
        t = _CpuUnpickler(f).load()
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32)
    t = t.float()
    print(f"  SASRec shape: {tuple(t.shape)}")
    return t


def load_node_embeddings(node_dir: str, csv_path: str) -> dict:
    """Returns {track_index: 1-D float tensor} for track_index < TOP_K."""
    df = pd.read_csv(csv_path)
    embeddings = {}
    missing = 0
    for _, row in df.iterrows():
        track_idx = int(row["track_index"])
        if track_idx >= TOP_K:
            continue
        fname = os.path.basename(row["embedding_path"])
        pt_path = os.path.join(node_dir, fname)
        if not os.path.exists(pt_path):
            missing += 1
            continue
        try:
            t = torch.load(pt_path, map_location="cpu", weights_only=False)
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, dtype=torch.float32)
            t = t.float().squeeze()
            if t.dim() > 1:
                t = t.mean(0)
            embeddings[track_idx] = t
        except Exception as e:
            print(f"  [WARN] Failed {pt_path}: {e}")
            missing += 1
    if missing:
        print(f"  [WARN] {missing} files missing/failed in {node_dir}")
    return embeddings


def load_one_pair(pair_idx: int):
    """Load audio + lyrics dicts for a single pair index (0-4)."""
    audio_node_name  = AUDIO_NODES[pair_idx]
    lyrics_node_name = LYRICS_NODES[pair_idx]

    audio_dir  = os.path.join(BATCH1_ROOT, "audio_embeddings",  audio_node_name)
    lyrics_dir = os.path.join(BATCH1_ROOT, "lyrics_embeddings", lyrics_node_name)

    print(f"  [{AUDIO_NAMES[pair_idx]}] loading from {audio_dir}")
    audio  = load_node_embeddings(audio_dir,  os.path.join(audio_dir,  "embeddings.csv"))
    print(f"  [{LYRICS_NAMES[pair_idx]}] loading from {lyrics_dir}")
    lyrics = load_node_embeddings(lyrics_dir, os.path.join(lyrics_dir, "embeddings.csv"))
    return audio, lyrics


def build_table(sources: list, max_idx: int, total_dim: int) -> torch.Tensor:
    """
    sources: list of dicts {track_index -> 1-D tensor}.
    Builds [max_idx+1, total_dim] by concatenating across sources per track.
    Tracks missing from any source get a zero row (they won't appear in training).
    """
    # common tracks present in all sources
    common = set(sources[0].keys())
    for s in sources[1:]:
        common &= set(s.keys())
    print(f"  Tracks present in all sources: {len(common)}")

    table = torch.zeros(max_idx + 1, total_dim, dtype=torch.float32)
    for tid in common:
        if tid > max_idx:
            continue
        parts = [s[tid] for s in sources]
        table[tid] = torch.cat(parts, dim=0)
    return table


def build(pair: int, include_sasrec: bool, data_path: str):
    sasrec = None
    if include_sasrec or pair == 5:
        if not os.path.exists(SASREC_PKL):
            raise FileNotFoundError(f"SASRec pkl not found: {SASREC_PKL}")
        sasrec = load_sasrec(SASREC_PKL)

    if pair == 5:
        # All 10 embedders + SASRec
        print("Pair 5: ALL embedders (CLAP+MERT+music2vec+EnCodec+MFCC + MiniLM+BGE-M3+MPNet+Multilingual+BERT) + SASRec")
        sources_dicts = []
        for i in range(5):
            a, l = load_one_pair(i)
            sources_dicts.append(a)
            sources_dicts.append(l)

        all_ids = set(sources_dicts[0].keys())
        for s in sources_dicts[1:]:
            all_ids &= set(s.keys())

        sample_id = next(iter(all_ids))
        audio_lyrics_dim = sum(s[sample_id].shape[0] for s in sources_dicts)
        sasrec_dim = sasrec.shape[1]
        total_dim = audio_lyrics_dim + sasrec_dim
        max_idx = TOP_K - 1
        print(f"  audio+lyrics dim: {audio_lyrics_dim}, SASRec dim: {sasrec_dim}, total: {total_dim}")

        table = torch.zeros(max_idx + 1, total_dim, dtype=torch.float32)
        for tid in all_ids:
            if tid > max_idx:
                continue
            al_parts = [s[tid] for s in sources_dicts]
            al_vec = torch.cat(al_parts, dim=0)
            sr_vec = sasrec[tid].float() if tid < sasrec.shape[0] else torch.zeros(sasrec_dim)
            table[tid] = torch.cat([al_vec, sr_vec], dim=0)

        out_name = "SASRec_item_embed_pair5_all_sasrec.pkl"

    else:
        print(f"Pair {pair}: {AUDIO_NAMES[pair]} (audio) + {LYRICS_NAMES[pair]} (lyrics)"
              + (" + SASRec" if include_sasrec else ""))
        audio, lyrics = load_one_pair(pair)

        all_ids = set(audio.keys()) & set(lyrics.keys())
        sample_id = next(iter(sorted(all_ids)))
        audio_dim  = audio[sample_id].shape[0]
        lyrics_dim = lyrics[sample_id].shape[0]
        total_dim  = audio_dim + lyrics_dim
        max_idx    = TOP_K - 1

        sources_dicts = [audio, lyrics]

        if include_sasrec:
            sasrec_dim = sasrec.shape[1]
            total_dim += sasrec_dim
            print(f"  audio dim: {audio_dim}, lyrics dim: {lyrics_dim}, SASRec dim: {sasrec_dim}, total: {total_dim}")
        else:
            print(f"  audio dim: {audio_dim}, lyrics dim: {lyrics_dim}, total: {total_dim}")

        table = torch.zeros(max_idx + 1, total_dim, dtype=torch.float32)
        for tid in all_ids:
            if tid > max_idx:
                continue
            parts = [audio[tid], lyrics[tid]]
            if include_sasrec:
                sr_vec = sasrec[tid].float() if tid < sasrec.shape[0] else torch.zeros(sasrec_dim)
                parts.append(sr_vec)
            table[tid] = torch.cat(parts, dim=0)

        suffix = "_sasrec" if include_sasrec else ""
        out_name = f"SASRec_item_embed_pair{pair}{suffix}.pkl"

    out_path = os.path.join(data_path, out_name)
    os.makedirs(data_path, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(table, f)
    print(f"\nSaved: {out_path}  shape={tuple(table.shape)}")
    return out_path, table.shape[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=int, required=True, choices=[0, 1, 2, 3, 4, 5],
                        help="0-4: single pair; 5: all 10 embedders + SASRec")
    parser.add_argument("--include_sasrec", action="store_true",
                        help="Concatenate SASRec collaborative embeddings for pairs 0-4")
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    args = parser.parse_args()

    out_path, dim = build(args.pair, args.include_sasrec, args.data_path)
    print(f"Done. dim={dim}")
