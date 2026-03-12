import gc
import os

import librosa
import numpy as np
import torch
import whisper
from transformers import (
    AutoModel, AutoProcessor,
    BertModel, BertTokenizer,
    ClapModel, ClapProcessor,
    EncodecModel,
    Wav2Vec2FeatureExtractor,
)
from sentence_transformers import SentenceTransformer

from paths import OUTPUT_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"

# 10 output .pt files — one per embedder
PT_FILES = {
    "audio_clap":        os.path.join(OUTPUT_DIR, "audio_clap.pt"),
    "audio_mert":        os.path.join(OUTPUT_DIR, "audio_mert.pt"),
    "audio_music2vec":   os.path.join(OUTPUT_DIR, "audio_music2vec.pt"),
    "audio_encodec":     os.path.join(OUTPUT_DIR, "audio_encodec.pt"),
    "audio_mfcc":        os.path.join(OUTPUT_DIR, "audio_mfcc.pt"),
    "text_minilm":       os.path.join(OUTPUT_DIR, "text_minilm.pt"),
    "text_bgem3":        os.path.join(OUTPUT_DIR, "text_bgem3.pt"),
    "text_mpnet":        os.path.join(OUTPUT_DIR, "text_mpnet.pt"),
    "text_multilingual": os.path.join(OUTPUT_DIR, "text_multilingual.pt"),
    "text_bert":         os.path.join(OUTPUT_DIR, "text_bert.pt"),
}

# .pt FILE HELPERS
def load_pt(path: str) -> dict:
    """Load an existing .pt store, or return a fresh empty one."""
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    return {
        "track_ids":    [],
        "artist_names": [],
        "track_names":  [],
        "embeddings":   None,
    }


def append_pt(path: str, track_id: int, artist: str, track: str,
              embedding: torch.Tensor):
    """
    Append one [dim] embedding row to a .pt store. Saves atomically.
    Uses .tmp → rename pattern to avoid corruption on crash.
    """
    store = load_pt(path)
    store["track_ids"].append(track_id)
    store["artist_names"].append(artist)
    store["track_names"].append(track)

    row = embedding.unsqueeze(0)   # [1, dim]
    store["embeddings"] = row if store["embeddings"] is None \
                          else torch.cat([store["embeddings"], row], dim=0)

    tmp = path + ".tmp"
    torch.save(store, tmp)
    os.replace(tmp, path)


def track_in_pt(path: str, track_id: int) -> bool:
    """Return True if track_id is already saved in this .pt file."""
    if not os.path.exists(path):
        return False
    store = torch.load(path, weights_only=False)
    return track_id in store["track_ids"]


# ─────────────────────────────────────────────────────────────────────────────
# CLAP HELPERS
# ClapProcessor in transformers 4.46.x requires audio as a list [audio_np].
# A bare numpy array is treated as None internally → raises ValueError.
# Both return-type variants handled (Tensor vs BaseModelOutputWithPooling).
# ─────────────────────────────────────────────────────────────────────────────

def _clap_audio_emb(model, proc, audio_np: np.ndarray) -> torch.Tensor:
    """Embed a 48kHz mono waveform with CLAP's audio tower → [512]."""
    inputs = proc(
        audios=[audio_np],       # list wrap required by ClapFeatureExtractor
        return_tensors="pt",
        sampling_rate=48000
    ).to(device)
    with torch.no_grad():
        out = model.get_audio_features(**inputs)
    tensor = out if isinstance(out, torch.Tensor) else out.pooler_output
    return tensor.squeeze(0).detach().cpu()


def _clap_text_emb(model, proc, text: str) -> torch.Tensor:
    """Embed a text string with CLAP's text tower → [512]."""
    inputs = proc(
        text=[text],
        return_tensors="pt",
        padding=True
    ).to(device)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
    tensor = out if isinstance(out, torch.Tensor) else out.pooler_output
    return tensor.squeeze(0).detach().cpu()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING — call once at startup, keep in memory for entire run
# ─────────────────────────────────────────────────────────────────────────────

def load_all_models() -> dict:
    """
    Load all 10 models + Whisper onto device.
    Returns a dict of model/processor references keyed by short name.
    Call this once in main — models stay in GPU memory for the entire run.
    """
    print(f"\n{'='*64}")
    print(f"  Loading all models onto {device.upper()}...")
    print(f"{'='*64}\n")

    m = {}

    # Whisper ASR — used for lyrics transcription
    print("  [ASR]  Whisper (base)...")
    m["whisper"] = whisper.load_model("small").to(device)

    # ── Audio embedders ───────────────────────────────────────────────────────

    # A1: LAION-CLAP | 512-dim | 48kHz
    # Paper: Wu et al., ICASSP 2023 — arxiv.org/abs/2211.06687
    print("  [A1]   LAION-CLAP (laion/larger_clap_music_and_speech) [512-dim]...")
    m["clap_proc"]  = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")
    m["clap_model"] = ClapModel.from_pretrained(
        "laion/larger_clap_music_and_speech"
    ).to(device)
    m["clap_model"].eval()

    # A2: MERT-v1-330M | 768-dim | 24kHz
    # Paper: Li et al., ICLR 2024 — arxiv.org/abs/2306.00107
    print("  [A2]   MERT-v1-330M (m-a-p/MERT-v1-330M) [768-dim]...")
    m["mert_proc"]  = Wav2Vec2FeatureExtractor.from_pretrained(
        "m-a-p/MERT-v1-330M", trust_remote_code=True
    )
    m["mert_model"] = AutoModel.from_pretrained(
        "m-a-p/MERT-v1-330M", trust_remote_code=True
    ).to(device)
    m["mert_model"].eval()

    # A3: Music2Vec-v1 | 768-dim | 16kHz
    # Paper: Li et al., ISMIR 2022 — arxiv.org/abs/2212.02508
    print("  [A3]   Music2Vec-v1 (m-a-p/music2vec-v1) [768-dim]...")
    m["m2v_proc"]  = Wav2Vec2FeatureExtractor.from_pretrained(
        "m-a-p/music2vec-v1", trust_remote_code=True
    )
    m["m2v_model"] = AutoModel.from_pretrained(
        "m-a-p/music2vec-v1", trust_remote_code=True
    ).to(device)
    m["m2v_model"].eval()

    # A4: Encodec-24kHz | 128-dim | 24kHz
    # Paper: Défossez et al., TMLR 2023 — arxiv.org/abs/2210.13438
    print("  [A4]   Encodec (facebook/encodec_24khz) [128-dim]...")
    m["encodec_proc"]  = AutoProcessor.from_pretrained("facebook/encodec_24khz")
    m["encodec_model"] = EncodecModel.from_pretrained(
        "facebook/encodec_24khz"
    ).to(device)
    m["encodec_model"].eval()

    # A5: MFCC via librosa | 128-dim (no model to load)
    # Replaces MusiCNN — incompatible with Python 3.10+ (requires numpy<1.17)
    print("  [A5]   MFCC via librosa (no model to load) [128-dim]...")

    # ── Text embedders ────────────────────────────────────────────────────────

    # T1: all-MiniLM-L6-v2 | 384-dim
    # Paper: Wang et al., NeurIPS 2020 — arxiv.org/abs/2002.10957
    print("  [T1]   all-MiniLM-L6-v2 [384-dim]...")
    m["minilm"] = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device=device
    )

    # T2: BAAI/bge-m3 | 1024-dim
    # Paper: Chen et al., 2024 — arxiv.org/abs/2402.03216
    print("  [T2]   BAAI/bge-m3 [1024-dim]...")
    m["bgem3"] = SentenceTransformer("BAAI/bge-m3", device=device)

    # T3: all-mpnet-base-v2 | 768-dim
    # Paper: Reimers & Gurevych, EMNLP 2019 — arxiv.org/abs/1908.10084
    print("  [T3]   all-mpnet-base-v2 [768-dim]...")
    m["mpnet"] = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2", device=device
    )

    # T4: paraphrase-multilingual-mpnet | 768-dim
    # Paper: Reimers & Gurevych, EMNLP 2020 — arxiv.org/abs/2004.09813
    # Critical for LastFM 1K — handles French, Korean, German lyrics
    print("  [T4]   paraphrase-multilingual-mpnet-base-v2 [768-dim]...")
    m["multilingual"] = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", device=device
    )

    # T5: bert-base-uncased | 768-dim (CLS token)
    # Paper: Devlin et al., NAACL 2019 — arxiv.org/abs/1810.04805
    print("  [T5]   bert-base-uncased [768-dim]...")
    m["bert_tok"]   = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    m["bert_model"] = BertModel.from_pretrained(
        "google-bert/bert-base-uncased"
    ).to(device)
    m["bert_model"].eval()

    print(f"\n  All models loaded.\n")
    return m


def unload_models(*items):
    """Free GPU memory by deleting model references."""
    for item in items:
        del item
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# EMBED ONE TRACK
# Accepts pre-loaded audio arrays — no disk reads inside this function.
# Audio must be loaded into RAM immediately after download (before cleanup).
# ─────────────────────────────────────────────────────────────────────────────

def embed_track(audio_48k: np.ndarray, audio_24k: np.ndarray,
                audio_16k: np.ndarray, lyrics: str, m: dict) -> dict:
    """
    Run all 10 embedders on pre-loaded audio arrays + lyrics string.

    Args:
        audio_48k: numpy array, 48kHz mono  (for CLAP)
        audio_24k: numpy array, 24kHz mono  (for MERT, Encodec)
        audio_16k: numpy array, 16kHz mono  (for Music2Vec, MFCC)
        lyrics:    transcribed lyrics string (from Whisper)
        m:         model dict returned by load_all_models()

    Returns:
        dict of { embedder_key: torch.Tensor [dim] }
    """
    embeddings = {}

    # A1: CLAP — 512-dim
    print(f"    [A1] CLAP audio (512-dim)...")
    embeddings["audio_clap"] = _clap_audio_emb(
        m["clap_model"], m["clap_proc"], audio_48k
    )

    # A2: MERT — 768-dim, mean-pool last hidden layer over time
    print(f"    [A2] MERT (768-dim)...")
    with torch.no_grad():
        inp = m["mert_proc"](
            audio_24k, sampling_rate=24000, return_tensors="pt"
        ).to(device)
        out = m["mert_model"](**inp, output_hidden_states=True)
        embeddings["audio_mert"] = (
            out.hidden_states[-1].mean(dim=1).squeeze(0).detach().cpu()
        )

    # A3: Music2Vec — 768-dim, mean-pool last hidden layer over time
    print(f"    [A3] Music2Vec (768-dim)...")
    with torch.no_grad():
        inp = m["m2v_proc"](
            audio_16k, sampling_rate=16000, return_tensors="pt"
        ).to(device)
        out = m["m2v_model"](**inp, output_hidden_states=True)
        embeddings["audio_music2vec"] = (
            out.hidden_states[-1].mean(dim=1).squeeze(0).detach().cpu()
        )

    # A4: Encodec — 128-dim, pre-VQ encoder states mean-pooled over time
    print(f"    [A4] Encodec (128-dim)...")
    with torch.no_grad():
        inp         = m["encodec_proc"](
            raw_audio=audio_24k, sampling_rate=24000, return_tensors="pt"
        ).to(device)
        encoder_out = m["encodec_model"].encoder(inp["input_values"])
        embeddings["audio_encodec"] = (
            encoder_out.mean(dim=-1).squeeze(0).detach().cpu()
        )

    # A5: MFCC — 128-dim
    # 40 MFCCs + deltas → mean + std over time → [160] → slice to [128]
    print(f"    [A5] MFCC (128-dim)...")
    mfcc  = librosa.feature.mfcc(y=audio_16k, sr=16000, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    embeddings["audio_mfcc"] = torch.tensor(
        np.concatenate([
            mfcc.mean(axis=1), mfcc.std(axis=1),
            delta.mean(axis=1), delta.std(axis=1),
        ]),
        dtype=torch.float32
    )[:128]

    # T1: MiniLM — 384-dim
    print(f"    [T1] MiniLM text (384-dim)...")
    embeddings["text_minilm"] = torch.tensor(
        m["minilm"].encode(lyrics), dtype=torch.float32
    )

    # T2: BGE-M3 — 1024-dim
    print(f"    [T2] BGE-M3 text (1024-dim)...")
    embeddings["text_bgem3"] = torch.tensor(
        m["bgem3"].encode(lyrics), dtype=torch.float32
    )

    # T3: all-mpnet — 768-dim
    print(f"    [T3] all-mpnet text (768-dim)...")
    embeddings["text_mpnet"] = torch.tensor(
        m["mpnet"].encode(lyrics), dtype=torch.float32
    )

    # T4: multilingual-mpnet — 768-dim
    print(f"    [T4] multilingual-mpnet text (768-dim)...")
    embeddings["text_multilingual"] = torch.tensor(
        m["multilingual"].encode(lyrics), dtype=torch.float32
    )

    # T5: BERT CLS token — 768-dim
    print(f"    [T5] BERT text CLS (768-dim)...")
    inp = m["bert_tok"](
        lyrics, return_tensors="pt",
        truncation=True, max_length=512, padding=True
    ).to(device)
    with torch.no_grad():
        out = m["bert_model"](**inp)
    embeddings["text_bert"] = (
        out.last_hidden_state[:, 0, :].squeeze(0).detach().cpu()
    )

    return embeddings