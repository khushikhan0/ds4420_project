import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import keras

from PIL import Image
from pathlib import Path
from coordinate_utils import get_sample_coordinates

"""
instructions to run the website:
1. make sure you're in the src directory
2. run this in terminal `streamlit run app.py`
"""

st.title('Wildfire Whackers')
st.write('Team Members: Ella Chee, Willbert Clement Christianto, Khushi Khan')

tab1, tab2, tab3 = st.tabs(["About Us", "Model Performance", "Interact With Data"])

with tab1:
    st.markdown('## Goal ##')
    st.write('Our project aims to detect the likelihood of wildfires using satellite wildfire imagery. We\'ve sampled nine images from our training dataset\
            below.')

    st.image('./website-visuals/dataset_samples.png')

    st.markdown('## Methodology ##')
    st.write('Our project implements MLP (Multi-layer Perceptron), CNN (Convolutional Neural Network) and Bayesian models to accomplish this binary classification task. \
            CNN and Bayesian modeling are an extension of the capabilities of the MLP model, as they introduce uncertainty quantification into the classification \
            task, complementing the deterministic MLP approach.')

    st.markdown('#### :gray[MLP Architecture] ####')
    st.write('The MLP is implemented manually in Python. The model maps an input vector (flattened image pixels) through a hidden layer (ReLU), and generates an output probability (Sigmoid). It’s trained using binary cross-entropy loss and optimized by gradient descent. This framework uses existing class resources, but carries out several adjustments to handle high-dimensional image data more efficiently. First, mini-batch gradient descent was used instead of full-batch updates to improve computational stability. The initialization technique was applied to the weights to prevent exploding gradients with ReLU activations, and input features were normalized using z-scoring to ensure consistent scaling. Lastly, vectorized operations are used to clip sigmoid inputs for numerical stability. These deviations extend the basic MLP framework from class to better suit large-scale image classification. ')

    st.markdown('#### :gray[CNN Architecture] ####')
    st.image('./cnn/cnn_architecture.png')
    st.write('The CNN is implemented using Keras and Tensorflow in Python. The model passes normalized data through a convolutional layer that applies ReLU, outputs 16 channels, then max pools the result with a (2,2) kernel. The second convolutional layer applies a ReLU function to the first pooled layer, outputting 32 channels. Max pooling is applied once more on the output of the second convolutional layer with a (2,2) kernel size. Each convolutional layer utilizes a (3,3) kernel with a stride of 1. Then, the pooled layer is flattened and passed through a hidden layer that applies ReLU, outputting 64 units. Finally, an output layer is defined with a Sigmoid activation function and an output space of 1 unit. The model is trained using an Adam (Adaptive Moment Estimation) optimizer, which trains in small batches of 32 images over 10 epochs. This optimizer is efficient on large datasets and adjusts learning rates dynamically during training. Finally, binary cross-entropy is used to calculate loss, as this task has a binary outcome. ')

    st.markdown('#### :gray[Bayesian Architecture] ####')
    st.write('The Bayesian logistic regression model is designed with brms in R, establishing a default prior with a Bernoulli distribution, and Bernoulli posterior, \
             as is standard for binary classification. 350x350 pixel images were resized to 35x35, RGB values were collapsed into a single value per pixel, and finally \
             flattened into a single vector such that each pixel represents an input feature (resulting in 1225 total features). Training was set up with four chains, \
             1000 iterations, and 180 burn-in iterations (18% of the initial samples).')

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
            "Precision": [0.88, 0.87, None, 0.88, 0.88],
            "Recall": [0.90, 0.85, None, 0.87, 0.88],
            "f1-score": [0.89, 0.86, 0.88, 0.87, 0.88],
            "Support": [3480, 2820, 6300, 6300, 6300],
        },
        index=["0.0", "1.0", "Accuracy", "Macro Avg", "Weighted Avg"],
    )

    st.table(classification_report)
    st.write('As seen in the classification report, the model has an F1-score of 0.89 for wildfire and 0.86 for no wildfire, indicating that \
            it is better at identifying wildfire-affected regions but the difference is immaterial to the scope/context of this project. Precision and recall \
            values are also closely aligned, delineating that there are no major biases toward false positives of false negatives.')
    st.image('./mlp/confusion_matrix_mlp.png')
    st.write('The following confusion matrix further supports our thesis: a majority of predictions fall along an accurate prediction with relatively few misclassifications \' \
            between wildfire and non-wildfire classes. This indicates that the model is able to distinguish visual patterns associated with wildfire presence despite being trained on \' \
            flattened image inputs. The mini-batch gradient descent and normalization techniques also support optimization stability that minimize overfitting. \
            Overall, while the MLP lacks the ability to explicitly capture spatial structure, it still achieves strong predictive capabilities. ')


    st.markdown('#### CNN Performance ####')
    st.write('The CNN was trained using Stochastic gradient descent. A binary cross entropy loss function was used. After 50 epochs of training with \
             a batch size of 32, the model\'s loss and accuracy converged to the following on the training and validation datasets:')
    st.image('./cnn/loss_plot_cnn.png')
    st.image('./cnn/accuracy_plot_cnn.png')
    st.write('The plots indicate that the model is learning the data well. This is especially shown in the Loss plot, as the loss on the validation set \
             converges to approximately 0.3. Additionally, the validation accuracy is high, oscillating between 88-94%.')
    st.markdown('\nThe model\'s performance was evaluated further using the test dataset.')
    st.image('./cnn/confusion_matrix_cnn.png')
    st.write('The CNN effectively captured spatial patterns in the satellite images. As shown by the confusion matrix, the model rarely predicted false negatives \
             and false positives. The model achieved an accuracy, precision, F1 score, and recall of approximately 95% on the test dataset. The high precision \
             indicates that the model is predicting wildfires accurately, while the high recall underlines that the model rarely labels images at risk of a wildfire \
             as the latter class. Obtaining a high recall score is especially valuable in wildfire detection, as false negatives can have a detrimental impact to human \
             life and the environment. The number of false negatives is lower than the number of false positives, which aligns well with this scenario. The F1 score \
             further demonstrates a good balance between precision and recall, suggesting robust overall performance. ')
    
    st.markdown('#### Bayesian Performance ####')
    st.write('The Bayesian logistic regression failed to meaningfully represent the distinction between areas satellite images of areas with and without risk of wildfire.\
             ')

with tab3:
    n_samples = 5000
    wildfire_lats, wildfire_longs = get_sample_coordinates(n_samples=n_samples)
    nowildfire_lats, nowildfire_longs = get_sample_coordinates(n_samples=n_samples, 
                                                               get_wildfire_set=False)

    st.markdown(f"### {n_samples} Image Samples from the Training Dataset")

    data = pd.DataFrame({
        'latitude': wildfire_lats + nowildfire_lats,
        'longitude': wildfire_longs + nowildfire_longs,
        'label': ['wildfire'] * n_samples + ['no_wildfire'] * n_samples
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
        model = keras.models.load_model('./cnn/cnn_model.keras')

        images = []
        coords = []
        for file in uploaded_files:
            coordinates = Path(file.name).stem.split(',')

            img = Image.open(file).convert("RGB")
            img = img.resize((256, 256))

            img_array = np.array(img)

            print(coordinates)

            images.append(img_array)
            coords.append((coordinates[0], coordinates[1]))

        images = np.array(images)

        preds = model.predict(images)

        print(preds)

        predicted_labels = (preds > 0.5).astype(int)

        for i, (label, img) in enumerate(zip(predicted_labels, images)):
            print(
                i,
                np.mean(img),
                np.std(img),
                img.shape
            )

            st.image(img, caption=f"({coordinates[0]}, {coordinates[1]})", width="content")
            st.write(f"Image {i + 1}: {'Wildfire Risk' if label == 1 else 'No Wildfire'}")
