import pandas as pd
import numpy as np
import os

from pathlib import Path
from PIL import Image

def get_sample_coordinates(get_wildfire_set=True):
    latitudes = []
    longitudes = []

    if get_wildfire_set:
        dataset_root = Path("test/wildfire")
    else:
        dataset_root = Path("test/nowildfire")

    all_files = list(Path(dataset_root).rglob("*.jpg")) + \
                list(Path(dataset_root).rglob("*.jpeg")) + \
                list(Path(dataset_root).rglob("*.png"))

    for path in all_files:
        file_path = Path(path)
        coordinates = file_path.stem
        long, lat = coordinates.split(',')
        latitudes.append(float(lat))
        longitudes.append(float(long))

    return latitudes, longitudes

def get_image_dict(get_wildfire_set=True):
    images = {}

    if get_wildfire_set:
        dataset_root = Path("test/wildfire")
    else:
        dataset_root = Path("test/nowildfire")

    all_files = list(Path(dataset_root).rglob("*.jpg")) + \
                list(Path(dataset_root).rglob("*.jpeg")) + \
                list(Path(dataset_root).rglob("*.png"))

    for path in all_files:
        file_path = Path(path)
        coordinates = file_path.stem
        images[coordinates] = path
    
    return images
        

if __name__ == "__main__":
    pass