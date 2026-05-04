'''
data_ingestion.py

Load dataset and configurations

'''

import os
import sys
import pandas as pd
import yaml

# Path setup
data_dir = "../Data"
third_party_dir = "../ThirdParty"
output_dir = "../Out"
cfg_dir = "../cfg"
sys.path.append(third_party_dir)

file_path = os.path.join(data_dir, "spotify_dataset.csv")
spotify_data_subset = os.path.join(data_dir, "spotify_dataset_500.csv")
CACHE_PATH = os.path.join(data_dir, "lyrics_cache.pkl")

from genius import Genius, save_lyrics, load_lyrics
# Add Lyric data to dataset
def _get_track_metadata(api_token=os.environ.get("GENIUS_ACCESS_TOKEN")):
    df = pd.read_csv(spotify_data_subset)
    client = Genius(api_token)

    embeddings = [None] * len(df)
    for index, row in df.iterrows():
        track = row["track_name"]
        artist = row["artists"]
        print(f"Index: {index}, Track: {track}, Artist: {artist}")
        try:
            embeddings[index] = client.embed(track, artist)
        except Exception as e:
            print(f"  skipped: {e}")

    df['lyric_embedding'] = embeddings
    return df

def get_track_data():
    all_data = pd.read_csv(file_path)
    
    tracks = all_data[["track_name", "artists"]]
    return tracks

def get_lyric_data(api_token=os.environ.get("GENIUS_ACCESS_TOKEN")):
    dataset_df = pd.read_csv(spotify_data_subset)
    client = Genius(api_token)
    
    cache = load_lyrics(CACHE_PATH) if os.path.exists(CACHE_PATH) else {}

    for index, row in dataset_df.iterrows():
        print(f"Index: {index}, Track: {row['track_name']}, Artist: {row['artists']}")
        key = (row['track_name'], row['artists'])
        if key in cache:
            continue  # already done
        try:
            cache[key] = client.lyrics(*key)
        except Exception as e:
            cache[key] = None  # mark as tried-and-failed so we don't retry
        if index % 100 == 0:
            save_lyrics(cache, CACHE_PATH)  # checkpoint

    save_lyrics(cache, CACHE_PATH)  # final save

def get_attributes():
    # Read and parse a YAML file
    with open(os.path.join(cfg_dir, "music_features.yaml"), 'r') as file:
        music_data = yaml.safe_load(file)

    features = music_data["audio_features"]
    music_elements = music_data["elements"]
    return features, music_elements

def get_feature_data():
    df = pd.read_csv(file_path)
    print("First 5 records:", df.head())
    
    features, music_elements = get_attributes()
    X = df[features].copy()
    X["explicit"] = df["explicit"].astype(int)
    return X

get_lyric_data()
# TODO:
# lyrical elements
# sentence segmentation