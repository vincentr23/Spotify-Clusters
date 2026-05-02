'''
Config.py

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

# Load the latest version
df = pd.read_csv(file_path)

# Read and parse a YAML file
with open(os.path.join(cfg_dir, "music_features.yaml"), 'r') as file:
    music_data = yaml.safe_load(file)

features = music_data["audio_features"]
music_elements = music_data["elements"]