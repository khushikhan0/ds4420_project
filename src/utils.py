from pathlib import Path
from PIL import Image, ImageFile
import numpy as np
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True

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

def plot_img(x, im_shape) -> None:
    plt.imshow(x.reshape(im_shape))
    plt.xticks([])
    plt.yticks([])
    plt.gcf().set_size_inches(4, 4)