import os

from PIL import Image
from pathlib import Path

def verify_images(directory):
    bad_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            path = os.path.join(directory, filename)
            try:
                with Image.open(path) as img:
                    img.verify()
            except (IOError, SyntaxError):
                bad_files.append(path)
                print(f"Corrupted: {path}")
    # Remove bad files
    for path in bad_files:
        os.remove(path)   

if __name__ == "__main__":
    data_root = Path("./data/raw")
    train_path = Path(data_root / "train")
    test_path = Path(data_root / "test")
    validation_path = Path(data_root / "valid")

    verify_images(train_path)
    verify_images(test_path)
    verify_images(validation_path)