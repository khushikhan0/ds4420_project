"""
Wildfire Prediction Models
Khushi Khan, Ella Chee, Willbert Clement Christianto
DS4420: Machine Learning and Data Mining 2
"""

import numpy as np
from PIL import Image

# loading current and existing data
existing_model = "mlp_weights.npz"
image_dimension = (32, 32)
classification = ["wildfire", "nowildfire"]

# activation functions, given binary classifier nature
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))

# pulling the existing model information
def load_model(path=existing_model):
    data = np.load(path, allow_pickle=True)
    return {"W1": data["W1"],
            "b1": data["b1"],
            "W2": data["W2"],
            "b2": data["b2"],
            "mean": data["mean"],
            "std": data["std"],
            }

# image preprocessing
def preprocess_image(image_path, mean, std):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(image_dimension)

    arr = np.array(img, dtype=np.float32) / 255.0

    # flatten to accommodate MLP
    arr = arr.flatten()
    arr = (arr - mean) / std

    return arr.reshape(1, -1)

def predict_output(X, model):
    W1 = model["W1"]
    b1 = model["b1"]
    W2 = model["W2"]
    b2 = model["b2"]

    return sigmoid(relu(X @ W1 + b1) @ W2 + b2)

def predict_image(image_path, model):
    X = preprocess_image(image_path, model["mean"], model["std"])
    prob = float(predict_output(X, model)[0, 0])

    pred_idx = int(prob >= 0.5)
    pred_label = classification[pred_idx]

    return {"predicted_class": pred_label,
            "probability": prob}

# example usage
if __name__ == "__main__":
    model = load_model()
    result = predict_image(".jpg", model)
    print(result)