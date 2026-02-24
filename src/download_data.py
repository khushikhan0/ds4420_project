import os
import kagglehub

os.environ["CACHE_HOME"] = os.path.join(os.getcwd(), "data/cache")

# Configuration
DATA_DIR = "data/"

# Utility
def create_data_dir():
    # Creating a data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

# Wildfire Images
def download_wildfire_data():
    # Downloading the wildfire images from dataset
    print("Downloading the wildfire images from Kaggle")

    kaggle_handle = "abdelghaniaaba/wildfire-prediction-dataset"
    path = kagglehub.dataset_download(handle=kaggle_handle, output_dir=DATA_DIR)
    
    print(f"Wildfire image dataset saved to {path}")

# Ensure the data directory exists and download wildfire data to it
if __name__ == "__main__":
    create_data_dir()
    download_wildfire_data()
    print("Data download complete.")
