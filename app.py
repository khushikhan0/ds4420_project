import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import keras

from PIL import Image
from pathlib import Path
from src.coordinate_utils import get_sample_coordinates
from src.mlp.mlp_test import predict_image, load_model

st.title('Wildfire Whackers')
st.write('Team Members: Ella Chee, Willbert Clement Christianto, Khushi Khan')

tab1, tab2, tab3, tab4 = st.tabs(["About Us", "Model Performance", "Data Visualizations", "View Our Poster"])

@st.cache_resource
def load_cnn():
    return keras.models.load_model('src/cnn/cnn_model.keras')

@st.cache_resource
def load_mlp():
    return load_model('src/mlp/mlp_weights.npz')

@st.cache_resource
def get_wildfire_samples():
    return get_sample_coordinates()

@st.cache_resource
def get_nowildfire_samples():
    return get_sample_coordinates(get_wildfire_set=False)

with tab1:
    st.markdown('## Goal ##')
    st.write('Our project aims to detect the likelihood of wildfires using satellite wildfire imagery. We\'ve sampled nine images from our training dataset\
            below.')

    st.image('src/website-visuals/dataset_samples.png')

    st.markdown('## Methodology ##')
    st.write('Our project implements MLP (Multi-layer Perceptron), CNN (Convolutional Neural Network) and Bayesian models to accomplish this binary classification task. \
            CNN and Bayesian modeling are an extension of the capabilities of the MLP model, as they introduce uncertainty quantification into the classification \
            task, complementing the deterministic MLP approach.')

    st.markdown('#### :gray[MLP Architecture] ####')
    st.write('The MLP is implemented manually in Python. The model maps an input vector (flattened image pixels) through a hidden layer (ReLU), and generates an output probability (Sigmoid). It’s trained using binary cross-entropy loss and optimized by gradient descent. This framework uses existing class resources, but carries out several adjustments to handle high-dimensional image data more efficiently. First, mini-batch gradient descent was used instead of full-batch updates to improve computational stability. The initialization technique was applied to the weights to prevent exploding gradients with ReLU activations, and input features were normalized using z-scoring to ensure consistent scaling. Lastly, vectorized operations are used to clip sigmoid inputs for numerical stability. These deviations extend the basic MLP framework from class to better suit large-scale image classification. ')

    st.markdown('#### :gray[CNN Architecture] ####')
    st.image('src/cnn/cnn_architecture.png')
    st.write('The CNN is implemented using Keras and Tensorflow in Python. The model passes normalized data through a convolutional layer that applies ReLU, outputs 16 channels, then max pools the result with a (2,2) kernel. The second convolutional layer applies a ReLU function to the first pooled layer, outputting 32 channels. Max pooling is applied once more on the output of the second convolutional layer with a (2,2) kernel size. Each convolutional layer utilizes a (3,3) kernel with a stride of 1. Then, the pooled layer is flattened and passed through a hidden layer that applies ReLU, outputting 64 units. Finally, an output layer is defined with a Sigmoid activation function and an output space of 1 unit. The model is trained using an Adam (Adaptive Moment Estimation) optimizer, which trains in small batches of 32 images over 10 epochs. This optimizer is efficient on large datasets and adjusts learning rates dynamically during training. Finally, binary cross-entropy is used to calculate loss, as this task has a binary outcome. ')

    st.markdown('#### :gray[Bayesian Architecture] ####')
    st.write('The Bayesian logistic regression model is designed with brms in R, establishing a default prior with a Bernoulli distribution, and Bernoulli posterior, as is standard for binary classification. 350x350px images were resized to 16x16px, RGB values were collapsed into a single value per pixel, and finally flattened into a single vector such that each pixel represents an input feature (resulting in 256 total features). Training was set up with four chains, 10000 iterations, and 1800 burn-in iterations (18% of the initial samples).')

    st.markdown('## Data Collection ##')
    st.write('The dataset is sourced from the Wildfire Prediction Dataset (Satellite Images) on \
            [Kaggle](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset/data), which contains 350x250 pixel satellite images of \
            Canadian wildfire-affected and non-affected regions. The dataset was split into train, test, and validation sets, with each set containing \
            directory for each respective class (wildfire, nowildfire). The visualization below describes the folder structure in more detail:')
    st.markdown("""
        ```bash
            ├───test
            │   ├───nowildfire
            │   └───wildfire
            ├───train
            │   ├───nowildfire
            │   └───wildfire
            └───valid
                ├───nowildfire
                └───wildfire
    """)

with tab2:
    st.markdown('#### MLP Performance ####')
    st.write('The MLP achieved stronger-than-anticipated results on the predicting wildfire presence. With a test accuracy >80% and balanced performance \
            across both classes.')

    classification_report = pd.DataFrame(
        {
            "Precision": [0.91, 0.89, None, 0.90, 0.90],
            "Recall": [0.91, 0.89, None, 0.90, 0.90],
            "f1-score": [0.91, 0.89, 0.90, 0.90, 0.90],
            "Support": [3480, 2820, 6300, 6300, 6300],
        },
        index=["0.0", "1.0", "Accuracy", "Macro Avg", "Weighted Avg"],
    )

    st.table(classification_report)
    st.write('As seen in the classification report, the model has an F1-score of 0.89 for wildfire and 0.86 for no wildfire, indicating that \
            it is better at identifying wildfire-affected regions but the difference is immaterial to the scope/context of this project. Precision and recall \
            values are also closely aligned, delineating that there are no major biases toward false positives of false negatives.')
    st.image('src/mlp/confusion_matrix_mlp.png')
    st.write('The following confusion matrix further supports our thesis: a majority of predictions fall along an accurate prediction with relatively few misclassifications \' \
            between wildfire and non-wildfire classes. This indicates that the model is able to distinguish visual patterns associated with wildfire presence despite being trained on \' \
            flattened image inputs. The mini-batch gradient descent and normalization techniques also support optimization stability that minimize overfitting. \
            Overall, while the MLP lacks the ability to explicitly capture spatial structure, it still achieves strong predictive capabilities. ')


    st.markdown('#### CNN Performance ####')
    st.write('The CNN was trained using Stochastic gradient descent. A binary cross entropy loss function was used. After 50 epochs of training with \
             a batch size of 32, the model\'s loss and accuracy converged to the following on the training and validation datasets:')
    st.image('src/cnn/loss_plot_cnn.png')
    st.image('src/cnn/accuracy_plot_cnn.png')
    st.write('The plots indicate that the model is learning the data well. This is especially shown in the Loss plot, as the loss on the validation set \
             converges to approximately 0.3. Additionally, the validation accuracy is high, oscillating between 88-94%.')
    st.markdown('\nThe model\'s performance was evaluated further using the test dataset.')
    st.image('src/cnn/confusion_matrix_cnn.png')
    st.write('The CNN effectively captured spatial patterns in the satellite images. As shown by the confusion matrix, the model rarely predicted false negatives \
             and false positives. The model achieved an accuracy, precision, F1 score, and recall of approximately 95% on the test dataset. The high precision \
             indicates that the model is predicting wildfires accurately, while the high recall underlines that the model rarely labels images at risk of a wildfire \
             as the latter class. Obtaining a high recall score is especially valuable in wildfire detection, as false negatives can have a detrimental impact to human \
             life and the environment. The number of false negatives is lower than the number of false positives, which aligns well with this scenario. The F1 score \
             further demonstrates a good balance between precision and recall, suggesting robust overall performance. ')
    
    st.markdown('#### Bayesian Performance ####')
    st.image('src/bayesian/brms_plot.png')
    st.image('src/bayesian/bayes_conf_matrix.png')
    st.image('src/bayesian/ppcheck.png')
    st.write('The Bayesian logistic regression failed to meaningfully represent the distinction between satellite images of areas with and without risk of wildfire. Decreasing the image size while increasing the number of iterations had some improvement, though various combinations of hyperparameters failed to reduce the r-hat to 1.00. In the final model, there were 2540 divergent transitions after warmup, with the largest r-hat being 7.52, indicated poor mixing of chains. As shown in the plots from a sample of features above, the posterior distribution fails to adequately mirror the Bernoulli, and rather than displaying random, hairy caterpillar-like chains across iterations, there seems to be little to no mixing of chains. Despite the poor convergence however, the model managed an overall accuracy of ~82%, and F1 score of 0.84, ranking slightly lower than the MLP.')

with tab3:
    wildfire_lats, wildfire_longs = get_wildfire_samples()
    nowildfire_lats, nowildfire_longs = get_nowildfire_samples()
    n_samples = len(wildfire_lats + wildfire_longs)

    st.markdown(f"### {n_samples} Image Samples from the Testing Dataset")

    data = pd.DataFrame({
        'latitude': wildfire_lats + nowildfire_lats,
        'longitude': wildfire_longs + nowildfire_longs,
        'label': ['wildfire'] * len(wildfire_lats) + ['no_wildfire'] * len(nowildfire_lats)
    })

    data['color'] = data['label'].map({
        'wildfire': [255, 0, 0], 
        'no_wildfire': [0, 255, 0]
    })

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position='[longitude, latitude]',
        get_color='color',
        get_radius=500,
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=data['latitude'].mean(),
        longitude=data['longitude'].mean(),
        zoom=4,
    )

    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state
    ))

    st.write('The red plot points indicate areas at risk of a wildfire. Green plot points are not at risk.')

    st.markdown("### Upload an image to test if a wildfire is likely: ###")

    uploaded_files = st.file_uploader(
        "Choose a file", accept_multiple_files=True
    )

    if uploaded_files:
        # Load models
        cnn_model = load_cnn()
        mlp_model = load_mlp()

        # Gather image and filename information
        images = []
        filenames = []
        for file in uploaded_files:
            split_file_name = Path(file.name).stem

            img = Image.open(file).convert("RGB")
            img = img.resize((256, 256))

            img_array = np.array(img)

            images.append(img_array)
            filenames.append(split_file_name)

        # Convert list of images to np array
        images = np.array(images)

        # Gathering CNN model's predictions
        cnn_preds = cnn_model.predict(images)
        predicted_labels = (cnn_preds > 0.5).astype(int)

        # Gathering MLP model's predictions
        mlp_preds = [predict_image(image, mlp_model)["predicted_class"] for image in uploaded_files]

        # Plot images and predictions from both models (CNN, MLP)
        for i, (label, img, filename, mlp_label) in enumerate(zip(predicted_labels, images, filenames, mlp_preds)):
            split_name = filename.split(',')

            if len(split_name) == 2:
                caption=f"({split_name[0]}, {split_name[1]})"
            else:
                caption=filename
            st.image(img, caption=caption, width="content")
            st.write(f"CNN Prediction: {'Wildfire Risk' if label == 1 else 'No Wildfire'}")
            st.write(f"MLP Prediction: {'Wildfire Risk' if mlp_label == 1 else 'No Wildfire'}")

with tab4:
    st.image('src/website-visuals/poster.png')