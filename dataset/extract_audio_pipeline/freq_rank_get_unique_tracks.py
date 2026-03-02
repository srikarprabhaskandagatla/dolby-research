import pandas as pd
import csv
from paths import file_path, output_file

def get_ordered_song_list(file_path):
    cols = ['userid', 'timestamp', 'artid', 'artist_name', 'traid', 'track_name']
    
    # FIX: Add quoting=csv.QUOTE_NONE to stop pandas from swallowing multi-line strings
    df = pd.read_csv(
        file_path, 
        sep='\t', 
        header=None, 
        names=cols, 
        on_bad_lines='skip',
        quoting=csv.QUOTE_NONE, # <-- This is the magic fix
        dtype=str
    )
    
    df.dropna(subset=['artist_name', 'track_name'], inplace=True)
    
    df = df[df['artist_name'].str.strip().astype(bool) & df['track_name'].str.strip().astype(bool)]
    
    # Strip whitespace from names
    df['artist_name'] = df['artist_name'].str.strip()
    df['track_name'] = df['track_name'].str.strip()
    
    # Replace problematic characters (commas, quotes)
    df['artist_name'] = df['artist_name'].str.replace(',', ';', regex=False).str.replace('"', '', regex=False)
    df['track_name'] = df['track_name'].str.replace(',', ';', regex=False).str.replace('"', '', regex=False)
    
    freq_table = df.groupby(['artist_name', 'track_name']).size().reset_index(name='play_count')
    
    # Sorting by frequency in descending order
    freq_table_sorted = freq_table.sort_values(by='play_count', ascending=False).reset_index(drop=True)
    
    return freq_table_sorted

ordered_songs_df = get_ordered_song_list(file_path)

ordered_songs_df.insert(0, 'track_index', range(len(ordered_songs_df)))
ordered_songs_df.to_csv(output_file, index=False)