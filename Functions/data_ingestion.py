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
def scrape_track_lyrics(api_token: str = None | os.environ.get("GENIUS_ACCESS_TOKEN")):
    df = pd.read_csv(file_path)
    
    for index, row in df.iterrows():
        track = row["track_name"]
        artist = row["artists"]
        try:
            lyrics = Genius(api_token).lyrics(track, artist)
        except Exception as e:
            print(f"Error fetching lyrics for '{track}' by '{artist}': {e}")


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