"""
embedder.py
───────────
Distributed embedding pipeline.  One embedder type per SLURM array task.

Each job (array index 0–9) loads a single model, scans all 10 data nodes
(50 000 tracks total), and writes embeddings to its own output sub-folder.

  EMBEDDER_ID  Name                  Modality  SR       Dim
  ───────────  ────────────────────  ────────  ──────   ────
  0            audio_clap            audio     48 kHz   512
  1            audio_mert            audio     24 kHz   768
  2            audio_music2vec       audio     16 kHz   768
  3            audio_encodec         audio     24 kHz   128
  4            audio_mfcc            audio     16 kHz   128
  5            text_minilm           text      —        384
  6            text_bgem3            text      —        1024
  7            text_mpnet            text      —        768
  8            text_multilingual     text      —        768
  9            text_bert             text      —        768

Input sources (all batch_1):
  Audio WAVs : /scratch3/.../raw_audio_files/batch_1/node_{N}/{track_id}.wav
  Status CSV : /scratch3/.../raw_audio_files/batch_1/download_status_node_{N}.csv
  Lyrics CSV : /scratch3/.../lyrics/batch_1/master_lyrics_node_{N}.csv

Output (per embedder):
  /scratch3/.../embeddings/node_{EMBEDDER_ID}/
    track_{id}/
      {emb_name}.pt
    embeddings.csv   ← track_index, artist_name, track_name, lyrics_source,
                         embedding_path

Usage:
  sbatch --array=0-9 embedder.sh
  # or manually:
  python embedder.py --embedder_id 3
"""

import argparse
import os
import sys
import traceback

import librosa
import numpy as np
import pandas as pd
import torch

# ── Embedder ID ────────────────────────────────────────────────────────────────

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--embedder_id",     type=int, default=None)
_parser.add_argument("--sub_node",        type=int, default=None,
                     help="Sub-node index (0-based) for parallel resume splits.")
_parser.add_argument("--total_sub_nodes", type=int, default=4,
                     help="Total number of sub-nodes to split remaining work across.")
_parser.add_argument("--resume_from",     type=int, default=None,
                     help="Resume from this track_id onward. Auto-detected from "
                          "existing CSVs if not set.")
_args, _ = _parser.parse_known_args()

if _args.embedder_id is not None:
    EMBEDDER_ID = _args.embedder_id
elif "SLURM_ARRAY_TASK_ID" in os.environ:
    EMBEDDER_ID = int(os.environ["SLURM_ARRAY_TASK_ID"])
else:
    EMBEDDER_ID = 0

SUB_NODE        = _args.sub_node         # None = no splitting (normal mode)
TOTAL_SUB_NODES = _args.total_sub_nodes
RESUME_FROM     = _args.resume_from      # None = auto-detect from max done track_id

# ── Embedder registry ──────────────────────────────────────────────────────────

EMBEDDER_MAP = {
    0: {"name": "audio_clap",        "modality": "audio", "sr": 48000},
    1: {"name": "audio_mert",        "modality": "audio", "sr": 24000},
    2: {"name": "audio_music2vec",   "modality": "audio", "sr": 16000},
    3: {"name": "audio_encodec",     "modality": "audio", "sr": 24000},
    4: {"name": "audio_mfcc",        "modality": "audio", "sr": 16000},
    5: {"name": "text_minilm",       "modality": "text"},
    6: {"name": "text_bgem3",        "modality": "text"},
    7: {"name": "text_mpnet",        "modality": "text"},
    8: {"name": "text_multilingual", "modality": "text"},
    9: {"name": "text_bert",         "modality": "text"},
}

if EMBEDDER_ID not in EMBEDDER_MAP:
    print(f"[ERROR] EMBEDDER_ID={EMBEDDER_ID} out of range (must be 0–9).")
    sys.exit(1)

EMB_NAME = EMBEDDER_MAP[EMBEDDER_ID]["name"]
MODALITY = EMBEDDER_MAP[EMBEDDER_ID]["modality"]
AUDIO_SR = EMBEDDER_MAP[EMBEDDER_ID].get("sr")   # None for text embedders

# ── Paths ──────────────────────────────────────────────────────────────────────

_AUDIO_BATCH_ROOT  = "/scratch3/workspace/skandagatla_umass_edu-dolby/raw_audio_files/batch_2"
_LYRICS_BATCH_ROOT = "/scratch3/workspace/skandagatla_umass_edu-dolby/lyrics/batch_2"
_OUTPUT_ROOT       = "/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings"

OUTPUT_DIR = os.path.join(_OUTPUT_ROOT, f"node_{EMBEDDER_ID}")

# Sub-node resume: each parallel worker writes its own CSV to avoid race conditions.
# Normal runs (no --sub_node) use the original embeddings.csv.
if SUB_NODE is not None:
    MANIFEST_CSV = os.path.join(OUTPUT_DIR, f"embeddings_sub_{SUB_NODE}.csv")
else:
    MANIFEST_CSV = os.path.join(OUTPUT_DIR, "embeddings.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

CHUNK_SEC = 30
device    = "cuda" if torch.cuda.is_available() else "cpu"

# ── File helpers ───────────────────────────────────────────────────────────────

def emb_path(track_id: int) -> str:
    return os.path.join(OUTPUT_DIR, f"{EMB_NAME}_{track_id}.pt")

def save_emb(track_id: int, tensor: torch.Tensor) -> None:
    """Atomic write via .tmp → rename so partial writes never corrupt."""
    path = emb_path(track_id)
    tmp  = path + ".tmp"
    torch.save(tensor.detach().cpu(), tmp)
    os.replace(tmp, path)

# ── CSV helpers ────────────────────────────────────────────────────────────────

def load_done_ids() -> tuple[set, int]:
    """
    Read all embeddings*.csv in OUTPUT_DIR.
    Returns (done_ids, max_track_id) where max_track_id is the highest
    track_index seen across all CSVs (used as the resume cutoff).
    """
    done = set()
    for fname in os.listdir(OUTPUT_DIR):
        if fname.startswith("embeddings") and fname.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(OUTPUT_DIR, fname))
                done.update(df["track_index"].tolist())
            except Exception:
                pass
    max_id = max(done) if done else 0
    return done, max_id

def append_manifest(row: dict) -> None:
    write_header = not os.path.exists(MANIFEST_CSV)
    pd.DataFrame([row]).to_csv(MANIFEST_CSV, mode="a", header=write_header, index=False)

# ── Data loading ───────────────────────────────────────────────────────────────

def load_all_tracks() -> pd.DataFrame:
    """
    Scan all 10 download_status CSVs.
    Returns DataFrame[track_index, data_node] for successfully downloaded tracks.
    """
    dfs = []
    for dn in range(10):
        csv_path = os.path.join(_AUDIO_BATCH_ROOT, f"download_status_node_{dn}.csv")
        if not os.path.exists(csv_path):
            print(f"  [WARN] Missing status CSV: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df["download_success"] = df["download_success"].astype(str).str.lower() == "true"
        ok = df[df["download_success"]][["track_index"]].copy()
        ok["track_index"] = ok["track_index"].astype(int)
        ok["data_node"] = dn
        dfs.append(ok)
    return (
        pd.concat(dfs, ignore_index=True)
        .drop_duplicates("track_index")
        .reset_index(drop=True)
    )

def load_all_lyrics() -> pd.DataFrame:
    """
    Combine all 10 master_lyrics CSVs.
    Returns DataFrame[track_index, artist_name, track_name,
                       lyrics_source, lyrics, detected_language].
    """
    dfs = []
    for dn in range(10):
        csv_path = os.path.join(_LYRICS_BATCH_ROOT, f"master_lyrics_node_{dn}.csv")
        if not os.path.exists(csv_path):
            print(f"  [WARN] Missing lyrics CSV: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df["track_index"] = df["track_index"].astype(int)
        dfs.append(df)
    return (
        pd.concat(dfs, ignore_index=True)
        .drop_duplicates("track_index")
        .reset_index(drop=True)
    )

# ── Audio chunking ─────────────────────────────────────────────────────────────

def _chunk_audio(audio: np.ndarray, sr: int, chunk_sec: int = CHUNK_SEC):
    chunk_len = sr * chunk_sec
    for start in range(0, len(audio), chunk_len):
        yield audio[start : start + chunk_len]

# ── Per-embedder compute functions ─────────────────────────────────────────────

def _embed_audio_clap(audio: np.ndarray, model, proc) -> torch.Tensor:
    chunks = []
    for chunk in _chunk_audio(audio, sr=48000):
        inputs = proc(audios=[chunk], return_tensors="pt", sampling_rate=48000).to(device)
        with torch.no_grad():
            out = model.get_audio_features(**inputs)
        t = out if isinstance(out, torch.Tensor) else out.pooler_output
        chunks.append(t.squeeze(0).detach().cpu())
        del inputs, out
    torch.cuda.empty_cache()
    return torch.stack(chunks).mean(dim=0)  # [512]


def _embed_audio_mert(audio: np.ndarray, model, proc) -> torch.Tensor:
    chunks = []
    min_samples = 24000  # 1 second minimum — MERT Conv1d requires non-trivial input
    for chunk in _chunk_audio(audio, sr=24000):
        if len(chunk) < min_samples:
            continue  # skip tail chunks too short for MERT's conv layers
        with torch.no_grad():
            inp = proc(chunk, sampling_rate=24000, return_tensors="pt").to(device)
            out = model(**inp, output_hidden_states=True)
            chunks.append(out.hidden_states[-1].mean(dim=1).squeeze(0).detach().cpu())
            del inp, out
    if not chunks:
        # entire audio was shorter than 1 s — process as-is and let MERT pad it
        with torch.no_grad():
            inp = proc(audio, sampling_rate=24000, return_tensors="pt").to(device)
            out = model(**inp, output_hidden_states=True)
            chunks.append(out.hidden_states[-1].mean(dim=1).squeeze(0).detach().cpu())
    torch.cuda.empty_cache()
    return torch.stack(chunks).mean(dim=0)  # [768]


def _embed_audio_music2vec(audio: np.ndarray, model, proc) -> torch.Tensor:
    chunks = []
    for chunk in _chunk_audio(audio, sr=16000):
        with torch.no_grad():
            inp = proc(chunk, sampling_rate=16000, return_tensors="pt").to(device)
            out = model(**inp, output_hidden_states=True)
            chunks.append(out.hidden_states[-1].mean(dim=1).squeeze(0).detach().cpu())
            del inp, out
    torch.cuda.empty_cache()
    return torch.stack(chunks).mean(dim=0)  # [768]


def _embed_audio_encodec(audio: np.ndarray, model, proc) -> torch.Tensor:
    chunks = []
    for chunk in _chunk_audio(audio, sr=24000):
        with torch.no_grad():
            inp = proc(raw_audio=chunk, sampling_rate=24000, return_tensors="pt").to(device)
            enc_out = model.encoder(inp["input_values"])
            chunks.append(enc_out.mean(dim=-1).squeeze(0).detach().cpu())
            del inp, enc_out
    torch.cuda.empty_cache()
    return torch.stack(chunks).mean(dim=0)  # [128]


def _embed_audio_mfcc(audio: np.ndarray) -> torch.Tensor:
    mfcc  = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    feat  = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        delta.mean(axis=1), delta.std(axis=1),
    ])
    return torch.tensor(feat, dtype=torch.float32)[:128]


def _embed_text_sentence(text: str, model) -> torch.Tensor:
    return torch.tensor(model.encode(text), dtype=torch.float32)


def _embed_text_bert(text: str, tok, model) -> torch.Tensor:
    inp = tok(
        text, return_tensors="pt",
        truncation=True, max_length=512, padding=True,
    ).to(device)
    with torch.no_grad():
        out = model(**inp)
    return out.last_hidden_state[:, 0, :].squeeze(0).detach().cpu()  # [768]

# ── Model loading (single model only) ─────────────────────────────────────────

def load_model() -> dict:
    """Load only the model required for this embedder ID."""
    m = {}
    print(f"  Loading {EMB_NAME!r} on {device.upper()}...")

    if EMBEDDER_ID == 0:
        from transformers import ClapModel, ClapProcessor
        m["proc"]  = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")
        m["model"] = ClapModel.from_pretrained("laion/larger_clap_music_and_speech").to(device).eval()

    elif EMBEDDER_ID == 1:
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        m["proc"]  = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        m["model"] = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True).to(device).eval()

    elif EMBEDDER_ID == 2:
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        m["proc"]  = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/music2vec-v1", trust_remote_code=True)
        m["model"] = AutoModel.from_pretrained("m-a-p/music2vec-v1", trust_remote_code=True).to(device).eval()

    elif EMBEDDER_ID == 3:
        from transformers import AutoProcessor, EncodecModel
        m["proc"]  = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        m["model"] = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device).eval()

    elif EMBEDDER_ID == 4:
        pass  # MFCC uses librosa only — no model to load

    elif EMBEDDER_ID == 5:
        from sentence_transformers import SentenceTransformer
        m["model"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    elif EMBEDDER_ID == 6:
        from sentence_transformers import SentenceTransformer
        m["model"] = SentenceTransformer("BAAI/bge-m3", device=device)

    elif EMBEDDER_ID == 7:
        from sentence_transformers import SentenceTransformer
        m["model"] = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)

    elif EMBEDDER_ID == 8:
        from sentence_transformers import SentenceTransformer
        m["model"] = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", device=device
        )

    elif EMBEDDER_ID == 9:
        from transformers import BertModel, BertTokenizer
        m["tok"]   = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
        m["model"] = BertModel.from_pretrained("google-bert/bert-base-uncased").to(device).eval()

    print("  Model ready.\n")
    return m

# ── Dispatch ───────────────────────────────────────────────────────────────────

def run_embedder(audio: np.ndarray | None, lyrics: str | None, m: dict) -> torch.Tensor:
    if   EMBEDDER_ID == 0: return _embed_audio_clap(audio, m["model"], m["proc"])
    elif EMBEDDER_ID == 1: return _embed_audio_mert(audio, m["model"], m["proc"])
    elif EMBEDDER_ID == 2: return _embed_audio_music2vec(audio, m["model"], m["proc"])
    elif EMBEDDER_ID == 3: return _embed_audio_encodec(audio, m["model"], m["proc"])
    elif EMBEDDER_ID == 4: return _embed_audio_mfcc(audio)
    elif EMBEDDER_ID == 9: return _embed_text_bert(lyrics, m["tok"], m["model"])
    else:                  return _embed_text_sentence(lyrics, m["model"])

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"{'=' * 64}")
    print(f"  Embedder {EMBEDDER_ID}  —  {EMB_NAME}")
    print(f"  Modality   : {MODALITY}" + (f"  @{AUDIO_SR} Hz" if AUDIO_SR else ""))
    print(f"  Output dir : {OUTPUT_DIR}")
    print(f"  Device     : {device.upper()}")
    if SUB_NODE is not None:
        print(f"  Sub-node   : {SUB_NODE} / {TOTAL_SUB_NODES}  (resume split)")
        print(f"  Manifest   : {MANIFEST_CSV}")
    print(f"{'=' * 64}\n")

    # ── Build full track list from all 10 data nodes ──────────────────────────
    print("Loading track index...")
    tracks_df = load_all_tracks()    # track_index, data_node

    print("Loading lyrics index...")
    lyrics_df = load_all_lyrics()    # track_index, artist_name, track_name, lyrics, ...

    merged = tracks_df.merge(
        lyrics_df[["track_index", "artist_name", "track_name", "lyrics_source", "lyrics"]],
        on="track_index",
        how="left",
    )

    total = len(merged)
    if total == 0:
        print("[WARN] No tracks found across all nodes.")
        return

    print(f"\nTotal tracks : {total}  "
          f"(IDs {merged['track_index'].min()} – {merged['track_index'].max()})")

    done_ids, max_done_id = load_done_ids()
    if done_ids:
        print(f"Resuming     : {len(done_ids)} tracks already done (max track_id={max_done_id}).")

    # Determine the cutoff: skip everything up to and including the last processed track.
    # This ignores skipped/failed tracks from the original run so we don't re-attempt them.
    cutoff = RESUME_FROM if RESUME_FROM is not None else max_done_id
    print(f"Cutoff       : track_index > {cutoff}")

    # Sort by track_index and keep only tracks strictly beyond the cutoff.
    remaining_df = (
        merged[merged["track_index"] > cutoff]
        .sort_values("track_index")
        .reset_index(drop=True)
    )

    if SUB_NODE is not None:
        # Contiguous split: divide remaining tracks into TOTAL_SUB_NODES sequential chunks.
        n      = len(remaining_df)
        size   = n // TOTAL_SUB_NODES
        start  = SUB_NODE * size
        end    = start + size if SUB_NODE < TOTAL_SUB_NODES - 1 else n  # last node gets remainder
        remaining_df = remaining_df.iloc[start:end].reset_index(drop=True)
        first_id = remaining_df["track_index"].iloc[0]  if len(remaining_df) else "—"
        last_id  = remaining_df["track_index"].iloc[-1] if len(remaining_df) else "—"
        print(f"Sub-node {SUB_NODE} slice : {len(remaining_df)} tracks  "
              f"(track_ids {first_id} – {last_id})\n")
    else:
        print(f"Remaining    : {len(remaining_df)}\n")

    if len(remaining_df) == 0:
        print("Nothing to do.")
        return

    # ── Load the single model for this embedder ───────────────────────────────
    model = load_model()

    # ── Process tracks ────────────────────────────────────────────────────────
    total = len(remaining_df)
    for idx, row in remaining_df.iterrows():
        track_id   = int(row["track_index"])
        data_node  = int(row["data_node"])
        artist     = str(row.get("artist_name", "")) if pd.notna(row.get("artist_name")) else ""
        track      = str(row.get("track_name",  "")) if pd.notna(row.get("track_name"))  else ""
        lyrics     = str(row["lyrics"])       if pd.notna(row.get("lyrics"))       else ""
        lyrics_src = str(row["lyrics_source"]) if pd.notna(row.get("lyrics_source")) else "unknown"

        print(f"\n[{idx+1}/{total}] track_{track_id} | {artist} — {track}")

        try:
            audio = None

            if MODALITY == "audio":
                wav_path = os.path.join(
                    _AUDIO_BATCH_ROOT, f"node_{data_node}", f"{track_id}.wav"
                )
                if not os.path.exists(wav_path):
                    print(f"  [WARN] WAV not found: {wav_path} — skipping.")
                    continue
                audio, _ = librosa.load(wav_path, sr=AUDIO_SR, mono=True)
                print(f"  Loaded audio  {len(audio)/AUDIO_SR:.1f}s  @{AUDIO_SR}Hz")

            else:  # text
                if not lyrics.strip():
                    print(f"  [WARN] No lyrics available — skipping.")
                    continue

            print(f"  Embedding with {EMB_NAME}...")
            emb = run_embedder(audio, lyrics, model)
            save_emb(track_id, emb)
            print(f"  Saved  shape={list(emb.shape)}  → track_{track_id}/{EMB_NAME}.pt")

            append_manifest({
                "track_index":    track_id,
                "artist_name":    artist,
                "track_name":     track,
                "lyrics_source":  lyrics_src,
                "embedding_path": emb_path(track_id),
            })

        except Exception as e:
            print(f"  [ERROR] track_{track_id}: {e}")
            traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print(f"Embedder {EMBEDDER_ID} ({EMB_NAME}) complete.")
    print(f"Manifest : {MANIFEST_CSV}")
    if os.path.exists(MANIFEST_CSV):
        df = pd.read_csv(MANIFEST_CSV)
        print(f"Tracks embedded : {len(df)}")


if __name__ == "__main__":
    main()
