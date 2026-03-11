import os
import pandas as pd
import torch
import whisper
from ytmusicapi import YTMusic
# from pytubefix import YouTube
import yt_dlp
from paths import *

# For the Genre
import requests
import urllib.parse

os.makedirs(output_freq_ordered_dir, exist_ok=True)
os.makedirs(temp_download_audio_dir, exist_ok=True)

ytmusic = YTMusic()

print("Loading Whisper ASR model...")
whisper_model = whisper.load_model("base") 

df = pd.read_csv(csv_freq_ordered_path)
metadata_log = []

for index, row in df.head(10).iterrows():
    track_id = row['track_index']
    artist = row['artist_name']
    track = row['track_name']
    
    final_embedding_path = os.path.join(output_freq_ordered_dir, f"track_{track_id}.pt")
    if os.path.exists(final_embedding_path):
        print(f"\nSkipping [{track_id}]: {artist} - {track} (Already embedded)")
        continue

    print(f"\nProcessing [{track_id}]: {artist} - {track}...")
    temp_audio_path = os.path.join(temp_download_audio_dir, f"{track_id}.m4a")
    
    try:
        # 1. SEARCH & DOWNLOAD
        search_query = f"{artist} {track}"
        search_results = ytmusic.search(search_query, filter="songs")
        
        if not search_results:
            search_results = ytmusic.search(search_query)

        if not search_results:
            print("Skipped: Could not find track on YouTube Music.")
            continue
            
        video_id = search_results[0]['videoId']
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Extract Length and Genre via pytubefix
        # yt = YouTube(video_url, client='TV', use_oauth=True, allow_oauth_cache=True)
        # song_length = yt.length 

        # General Notation
        # genre = "Unknown"
        # try:
        #     if yt.metadata and len(yt.metadata) > 0:
        #         genre = yt.metadata[0].get('Genre', 'Unknown')
        #     if genre == "Unknown" and yt.keywords:
        #         genre = ", ".join(yt.keywords[:3]) 
        # except Exception:
        #     pass

        # EXTRACT GENRE VIA ITUNES API (No Auth Required)
        genre = "Unknown"
        try:
            # Format the query safely for a URL (e.g. "Radiohead Karma Police")
            query = urllib.parse.quote(f"{artist} {track}")
            itunes_url = f"https://itunes.apple.com/search?term={query}&entity=song&limit=1"
            
            response = requests.get(itunes_url, timeout=5).json()
            
            if response['resultCount'] > 0:
                genre = response['results'][0].get('primaryGenreName', 'Unknown')
        except Exception as e:
            print(f"Genre lookup failed: {e}")
        
        # Download Audio
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(temp_download_audio_dir, f"{track_id}.%(ext)s"),
            'cookiefile': 'cookies.txt',
            
            'extractor_args': {
                'youtube': {
                    'client': ['android_music', 'android', 'web']
                }
            },
            
             'js_runtimes': {'node': {}},

            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'm4a',
            }],
            
            'quiet': True,
            'no_warnings': True
        }
        
        print("Downloading audio stream...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # yt-dlp extracts the duration and downloads the file simultaneously
            info_dict = ydl.extract_info(video_url, download=True)
            song_length = info_dict.get('duration', 0)
            
        print(f"Downloaded audio to {temp_audio_path}")

        # 2. WHISPER ASR (Lyrics Extraction)
        print("Running Whisper ASR transcription...")
        transcription_result = whisper_model.transcribe(temp_audio_path)
        lyrics_text = transcription_result["text"].strip()
        
        print_lyrics = f"{lyrics_text[:50]}..." if lyrics_text else "No speech detected"
        print(f"Transcription complete: {print_lyrics}")

        # 3. K + N EMBEDDING GENERATION
        print("Generating k audio and n text embeddings...")
        
        # -> INSERT ACTUAL MODEL INFERENCE HERE <-
        # audio_emb_clap = audio_model.encode(temp_audio_path)
        # text_emb_bge = text_model.encode(lyrics_text)
        
        # Simulating the generated vectors with random tensors for the pipeline structure
        audio_emb_1 = torch.rand(768)  # Dummy CLAP/BEATs vector
        audio_emb_2 = torch.rand(1024) # Dummy MERT vector
        
        text_emb_1 = torch.rand(768)   # Dummy BGE-M3 vector
        text_emb_2 = torch.rand(384)   # Dummy all-MiniLM vector
        
        # 4. STORE EMBEDDINGS
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
                "model_k1_name": audio_emb_1,
                "model_k2_name": audio_emb_2
            },
            "text_embeddings": {
                "model_n1_name": text_emb_1,
                "model_n2_name": text_emb_2
            }
        }
        
        torch.save(track_data, final_embedding_path)
        print(f"Success! Saved multidimensional embeddings to {final_embedding_path}")
        
        metadata_log.append(track_data["metadata"])
        
    except Exception as e:
        print(f"Failed to process {track_id}: {e}")
        
    finally:
        # 5. CLEANUP (Delete the raw audio)
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            print(f"Deleted temp file: {temp_audio_path}")

# Final Logging
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