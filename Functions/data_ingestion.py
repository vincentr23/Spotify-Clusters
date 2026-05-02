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

from genius import Genius
# Add Lyric data to dataset
def get_track_data(api_token=os.environ.get("GENIUS_ACCESS_TOKEN")):
    df = pd.read_csv(file_path)
    client = Genius(api_token)

    df['lyric_embedding'] = pd.Series([None] * len(df), dtype=object)

    for index, row in df.iterrows():
        print(f"Index: {index}, Track: {row['track_name']}, Artist: {row['artists']}")

        track = row["track_name"]
        artist = row["artists"]
        try:
            lyrics = client.lyrics(track, artist)
            vec = client.embed(track, artist)
        except Exception as e:
            vec = None
            print(f"Error fetching lyrics for '{track}' by '{artist}': {e}")
        df.at[index, 'lyric_embedding'] = vec

    return df

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

# TODO:
# lyrical elements
# sentence segmentation