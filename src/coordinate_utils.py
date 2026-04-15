import pandas as pd
import numpy as np
import os

from pathlib import Path
from PIL import Image

def get_sample_coordinates(n_samples=100, get_wildfire_set=True):
    latitudes = []
    longitudes = []

    if get_wildfire_set:
        dataset_root = Path("../data/raw/train/wildfire")
    else:
        dataset_root = Path("../data/raw/train/nowildfire")

    all_files = list(Path(dataset_root).rglob("*.jpg")) + \
                list(Path(dataset_root).rglob("*.jpeg")) + \
                list(Path(dataset_root).rglob("*.png"))

    for path in all_files:
        file_path = Path(path)
        coordinates = file_path.stem
        long, lat = coordinates.split(',')
        latitudes.append(float(lat))
        longitudes.append(float(long))

    sample_size = min(n_samples, len(latitudes))

    return latitudes[:sample_size], longitudes[:sample_size]

if __name__ == "__main__":
    pass