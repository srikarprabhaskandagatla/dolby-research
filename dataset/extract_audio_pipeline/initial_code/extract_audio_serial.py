import pandas as pd
import os
from ytmusicapi import YTMusic
from pytubefix import YouTube
from paths import csv_path, output_dir

ytmusic = YTMusic()

os.makedirs(output_dir, exist_ok=True)
df = pd.read_csv(csv_path)

metadata_log = []

for index, row in df.head(200).iterrows():
    track_id = row['track_index']
    artist = row['artist_name']
    track = row['track_name']
    
    print(f"\nProcessing [{track_id}]: {artist} - {track}...")
    
    try:
        search_query = f"{artist} {track}"
        search_results = ytmusic.search(search_query, filter="songs")
        
        if not search_results:
            search_results = ytmusic.search(search_query)

        if search_results:
            video_id = search_results[0]['videoId']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            
            # EXTRACT LYRICS VIA YTMUSICAPI
            lyrics_text = "Not Found"
            try:
                watch_playlist = ytmusic.get_watch_playlist(videoId=video_id)
                if 'lyrics' in watch_playlist and watch_playlist['lyrics']:
                    lyrics_id = watch_playlist['lyrics']
                    lyrics_data = ytmusic.get_lyrics(lyrics_id)
                    lyrics_text = lyrics_data.get('lyrics', 'Not Found')
            except Exception as e:
                pass 
            
            # EXTRACT LENGTH AND GENRE VIA PYTUBEFIX
            yt = YouTube(video_url, client='TV', use_oauth=True, allow_oauth_cache=True)
            
            song_length = yt.length 
            
            genre = "Unknown"
            try:
                if yt.metadata and len(yt.metadata) > 0:
                    genre = yt.metadata[0].get('Genre', 'Unknown')
                
                if genre == "Unknown" and yt.keywords:
                    genre = ", ".join(yt.keywords[:3]) 
            except Exception:
                pass 
            
            # DOWNLOAD AUDIO
            audio_stream = yt.streams.get_audio_only()
            audio_stream.download(output_path=output_dir, filename=f"{track_id}.m4a")
            
            metadata_log.append({
                'track_index': track_id,
                'artist_name': artist,
                'track_name': track,
                'duration_seconds': song_length,
                'youtube_genre_or_tags': genre,
                'lyrics': lyrics_text,
                'youtube_video_id': video_id
            })
            
            print_lyrics = f"{lyrics_text[:50]}..." if lyrics_text != "Not Found" else "Not Found"
            print(f"Success! Saved {track_id}.m4a | Length: {song_length}s | Genre: {genre} | Lyrics: {print_lyrics}")
            
        else:
            print(f"Skipped: Could not find track on YouTube Music.")
            
    except Exception as e:
        print(f"Failed to process: {e}")

if metadata_log:
    metadata_df = pd.DataFrame(metadata_log)
    metadata_output_path = os.path.join(output_dir, "downloaded_metadata.csv")
    metadata_df.to_csv(metadata_output_path, index=False)
    print(f"\nBatch complete. Metadata saved to {metadata_output_path}")
else:
    print("\nBatch complete. No tracks were successfully downloaded.")