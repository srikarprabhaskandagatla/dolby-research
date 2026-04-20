import os
import pandas as pd
import torch
import whisper
import librosa
import requests
import urllib.parse
from ytmusicapi import YTMusic
from pytubefix import YouTube
from transformers import ClapModel, ClapProcessor, Wav2Vec2FeatureExtractor, AutoModel
from sentence_transformers import SentenceTransformer
from paths import *

# --- Configuration & Paths ---
os.makedirs(output_freq_ordered_dir, exist_ok=True)
os.makedirs(temp_download_audio_dir, exist_ok=True)

# Automatically map to Unity's GPUs if available, otherwise fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Executing pipeline on device: {device.upper()}")

# --- Initialization & Model Loading ---
ytmusic = YTMusic()

print("Loading Whisper ASR model...")
whisper_model = whisper.load_model("base").to(device)

print("Loading Audio Embedders (k=2)...")
# Audio Model 1: CLAP (Multimodal semantic audio)
clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
clap_model.eval()

# Audio Model 2: MERT (Music Information Retrieval specific)
mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True).to(device)
mert_model.eval()

print("Loading Text Embedders (n=2)...")
# Text Model 1: all-MiniLM (Fast, baseline semantics)
minilm_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Text Model 2: BGE-M3 (State-of-the-art multilingual)
bgem3_model = SentenceTransformer('BAAI/bge-m3', device=device)

print("All models loaded successfully into VRAM.\n")

# --- Pipeline Execution ---
df = pd.read_csv(csv_freq_ordered_path)
metadata_log = []

for index, row in df.head(10).iterrows():
    track_id = row['track_index']
    artist = row['artist_name']
    track = row['track_name']
    
    # Checkpoint: Skip if embedding already exists
    final_embedding_path = os.path.join(output_freq_ordered_dir, f"track_{track_id}.pt")
    if os.path.exists(final_embedding_path):
        print(f"\nSkipping [{track_id}]: {artist} - {track} (Already embedded)")
        continue

    print(f"\nProcessing [{track_id}]: {artist} - {track}...")
    temp_audio_path = os.path.join(temp_download_audio_dir, f"{track_id}.m4a")
    
    try:
        # ---------------------------------------------------
        # 1. SEARCH & DOWNLOAD
        # ---------------------------------------------------
        search_query = f"{artist} {track}"
        search_results = ytmusic.search(search_query, filter="songs")
        
        if not search_results:
            search_results = ytmusic.search(search_query)

        if not search_results:
            print("Skipped: Could not find track on YouTube Music.")
            continue
            
        video_id = search_results[0]['videoId']
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        yt = YouTube(video_url, client='TV', use_oauth=True, allow_oauth_cache=True)
        song_length = yt.length 

        # ---------------------------------------------------------
        # EXTRACT GENRE VIA ITUNES API
        # ---------------------------------------------------------
        genre = "Unknown"
        try:
            query = urllib.parse.quote(f"{artist} {track}")
            itunes_url = f"https://itunes.apple.com/search?term={query}&entity=song&limit=1"
            response = requests.get(itunes_url, timeout=5).json()
            if response['resultCount'] > 0:
                genre = response['results'][0].get('primaryGenreName', 'Unknown')
        except Exception as e:
            print(f"Genre lookup failed: {e}")
        
        # Download Audio
        audio_stream = yt.streams.get_audio_only()
        audio_stream.download(output_path=temp_download_audio_dir, filename=f"{track_id}.m4a")
        print(f"Downloaded audio to {temp_audio_path}")

        # ---------------------------------------------------
        # 2. WHISPER ASR (Lyrics Extraction)
        # ---------------------------------------------------
        print("Running Whisper ASR transcription...")
        transcription_result = whisper_model.transcribe(temp_audio_path)
        lyrics_text = transcription_result["text"].strip()
        
        # If the track is entirely instrumental, ensure we pass at least an empty string 
        # to the text embedders to avoid crash
        if not lyrics_text:
            lyrics_text = "[Instrumental]"
            
        print_lyrics = f"{lyrics_text[:50]}..."
        print(f"Transcription complete: {print_lyrics}")

        # ---------------------------------------------------
        # 3. K + N EMBEDDING GENERATION
        # ---------------------------------------------------
        print("Generating tensor representations...")
        
        with torch.no_grad(): # Disable gradients to save memory
            # --- Text Embeddings (n=2) ---
            emb_text_minilm = torch.tensor(minilm_model.encode(lyrics_text))
            emb_text_bgem3 = torch.tensor(bgem3_model.encode(lyrics_text))

            # --- Audio Embeddings (k=2) ---
            # CLAP expects 48kHz sample rate
            audio_48k, _ = librosa.load(temp_audio_path, sr=48000)
            clap_inputs = clap_processor(audios=audio_48k, return_tensors="pt", sampling_rate=48000).to(device)
            emb_audio_clap = clap_model.get_audio_features(**clap_inputs).cpu().squeeze()
            
            # MERT expects 24kHz sample rate
            audio_24k, _ = librosa.load(temp_audio_path, sr=24000)
            mert_inputs = mert_processor(audio_24k, sampling_rate=24000, return_tensors="pt").to(device)
            mert_outputs = mert_model(**mert_inputs, output_hidden_states=True)
            emb_audio_mert = mert_outputs.hidden_states[-1].mean(dim=1).cpu().squeeze()
        
        # ---------------------------------------------------
        # 4. STORE EMBEDDINGS
        # ---------------------------------------------------
        track_data = {
            "metadata": {
                "track_index": track_id,
                "artist_name": artist,
                "track_name": track,
                "duration_seconds": song_length,
                "youtube_genre_or_tags": genre,
                "youtube_video_id": video_id,
                "whisper_lyrics": lyrics_text
            },
            "audio_embeddings": {
                "clap_htsat_unfused": emb_audio_clap,
                "mert_v1_330M": emb_audio_mert
            },
            "text_embeddings": {
                "all_minilm_l6_v2": emb_text_minilm,
                "bge_m3": emb_text_bgem3
            }
        }
        
        torch.save(track_data, final_embedding_path)
        print(f"Success! Saved multidimensional embeddings to {final_embedding_path}")
        
        metadata_log.append(track_data["metadata"])
        
    except Exception as e:
        print(f"Failed to process {track_id}: {e}")
        
    finally:
        # ---------------------------------------------------
        # 5. CLEANUP (Delete the raw audio)
        # ---------------------------------------------------
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            print(f"Deleted temp file: {temp_audio_path}")

# --- Final Logging ---
if metadata_log:
    metadata_df = pd.DataFrame(metadata_log)
    metadata_output_path = os.path.join(output_freq_ordered_dir, "processed_metadata_log.csv")
    
    if os.path.exists(metadata_output_path):
        metadata_df.to_csv(metadata_output_path, mode='a', header=False, index=False)
    else:
        metadata_df.to_csv(metadata_output_path, index=False)
        
    print(f"\nBatch complete. Metadata logged to {metadata_output_path}")
else:
    print("\nBatch complete. No tracks were successfully processed.")