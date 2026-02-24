import pandas as pd

file_path = "[INPUT_FILE_PATH]"
output_file = "[OUTPUT_FILE_PATH]"

raw_columns = [
    "userid", "timestamp", "musicbrainz-artist-id", 
    "artist-name", "musicbrainz-track-id", "track-name"
]

cols_to_use = ["artist-name", "track-name", "musicbrainz-track-id"]

df = pd.read_csv(
    file_path, 
    sep='\t', 
    header=None, 
    names=raw_columns, 
    usecols=cols_to_use, 
    on_bad_lines='skip'
)

print("Data loaded successfully!")

df = df.dropna(subset=['track-name', 'artist-name'])

unique_songs = df.drop_duplicates()

unique_songs = unique_songs.rename(columns={
    "artist-name": "artist_name",
    "track-name": "track_name",
    "musicbrainz-track-id": "musicbrainz_track_id"
})

unique_songs = unique_songs[["artist_name", "track_name", "musicbrainz_track_id"]]

# Add the requested 'track_index' column starting from 0
unique_songs.insert(0, 'track_index', range(len(unique_songs)))

unique_songs.to_csv(output_file, index=False)

print(f"Successfully completed! - No. of unique tracks: {len(unique_songs)}")