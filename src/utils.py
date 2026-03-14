import cv2
from pathlib import Path
from PIL import Image, ImageFile
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

def convert_to_grayscale(path: str) -> None:
    '''
    Converts an image(s) at the given path to grayscale.
    '''
    input_path = Path(path)

    # Determines if path was a file path or folder path
    if input_path.is_file():
        files = [input_path]
        output_dir = input_path.parent / (input_path.parent.name + "_grey")
    elif input_path.is_dir():
        files = [f for f in input_path.iterdir() if f.suffix.lower() == ".jpg"]
        output_dir = input_path / (input_path.name + "_grey")
    else:
        print(f"Path not found: {path}")
        return

    # Don't make a directory if there are no files
    if not files:
        return
    # Ensures the output directory exists
    output_dir.mkdir(exist_ok=True)

    # Iterates over every file and converts it to grayscale
    for file in files:
        image = cv2.imread(str(file))
        if image is None:
            print(f"File not found or cannot be read: {file}")
            continue

        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        save_path = output_dir / file.name

        # Save grayscale file to save path
        cv2.imwrite(str(save_path), gray_img)

    print(f"Grayscale images saved to: {output_dir}")

def convert_to_matrix(image_path: str) -> np.array:
    im = Image.open(image_path)
    return np.array(im)

def folder_to_matrices(folder_path: str, img_dim: tuple) -> list[np.ndarray]:
    folder = Path(folder_path)
    matrices = []
    for file in folder.iterdir():
        if file.suffix.lower() == ".jpg":
            img = Image.open(file).convert("RGB").resize(img_dim)
            matrices.append(np.array(img, dtype=np.float32) / 255.0)
    return np.stack(matrices)

def split_to_matrices(split_path: str, img_dim: tuple, classes: list[str]) -> np.ndarray:
    X, y = [], []
    for label, folder_path in enumerate(classes):
        class_matrices = folder_to_matrices(Path(split_path) / folder_path, img_dim)
        X.append(class_matrices)
        y.append(np.full(len(class_matrices), label))
    return np.concatenate(X), np.concatenate(y)