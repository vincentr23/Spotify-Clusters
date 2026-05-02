'''
data_ingestion.py

Load dataset and configurations

'''

import os 
import pandas as pd
import yaml

# Path setup
data_dir = "../Data"
third_party_dir = "../ThirdParty"
output_dir = "../Out"
cfg_dir = "../cfg"
file_path = os.path.join(data_dir, "spotify_dataset.csv")

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