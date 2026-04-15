import streamlit as st
import pandas as pd
import numpy as np

from coordinate_utils import get_sample_coordinates

st.title('Wildfire Whackers')

st.write('Team Members: Ella Chee, Willbert Clement Christianto, Khushi Khan')

tab1, tab2, tab3 = st.tabs(["About Us", "Model Performance", "Interact with data"])

with tab1:
    st.markdown('### Project Goal ###')
    st.write('Our project aims to detect the likelihood of wildfires using satellite wildfire imagery. We\'ve sampled nine images from our training dataset\
            below.')

    st.image('./website-visuals/dataset_samples.png')

    st.markdown('### Methodology ###')
    st.write('Our project implements MLP (Multi-layer Perceptron), CNN (Convolutional Neural Network) and Bayesian models to accomplish this classification task. \
            CNN and Bayesian modeling are an extension of the capabilities of the MLP model, as they introduce uncertainty quantification into the classification \
            task, complementing the deterministic MLP approach.')

    st.markdown('#### :gray[MLP Architecture] ####')
    st.write('The MLP model maps an input vector (flattened image pixels) through a hidden layer (ReLU), and generates an output probability (Sigmoid). \
                The model is trained using binary cross-entropy loss and optimized with the help of gradient descent. This framework uses existing class resources, \
                but carries out several adjustments designed to handle high-dimensional image data more efficiently. First of all, mini-batch gradient descent was \
                used instead of full-batch updates to improve computational stability. The initialization technique was applied to the weights to prevent exploding \
                gradients with ReLU activations, and input features were normalized using z-scoring to ensure consistent scaling. Last but not least, the \
                implementation uses vectorized operations to clip sigmoid inputs to maintain numerical stability. These deviations extend the basic MLP framework \
                from class to better suit large-scale image classification tasks like this.')

    st.markdown('#### :gray[CNN Architecture] ####')
    st.write('The model passes normalized data through a convolutional layer that outputs 16 channels, then applies max pooling. The second convolutional layer \
            applies a Sigmoid activation function to the first pooled layer, outputting 32 channels. Max pooling is applied once more on the output of the second \
            convolutional layer. Each convolutional and pool layer utilizes a (3, 3) kernel with a stride of 1. Then, the pooled layer is flattened and passed \
            through a hidden layer that applies a ReLU activation function. The dimensionality of the output space is 250 units. Finally, an output layer is \
            defined with a Sigmoid activation function and an output space of 1 unit. The model outputs either a 0 (not at risk of a wildfire) or a 1 (at risk \
            of a wildfire), which is why only 1 unit is needed for this layer. The model is trained using a Stochastic Gradient Descent optimizer, which trains \
            in small batches of 32 images over 50 epochs. This optimizer was chosen to account for the large dataset size. Finally, binary cross-entropy is used \
            to calculate loss, as this task has a binary outcome.')

    st.markdown('#### :gray[Bayesian Architecture] ####')
    st.write('The model ...')

    st.markdown('### Data Collection ###')
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
    st.markdown("""
        ```bash
                            Classification Report:
                        precision    recall  f1-score   support

                    0.0       0.88      0.90      0.89      3480
                    1.0       0.87      0.85      0.86      2820

                accuracy                           0.88      6300
            macro avg       0.88      0.87      0.87      6300
            weighted avg       0.88      0.88      0.88      6300
    """)
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
    st.write('The plots indicate that the model is learning the data well. This is especially shown in the Loss plot, as the loss on the validation set\
             converges between 0.10 and 0.15. Additionally, the validation accuracy is high, oscillating between 95-97%.')
    st.markdown('\nThe model\'s performance was evaluated further using the test dataset.')
    st.image('./cnn/confusion_matrix_cnn.png')
    st.write('The CNN effectively captured spatial patterns in the wildfire images. As shown by the confusion matrix, the model rarely predicted false \
             negatives and false positives. The model achieved an accuracy, precision, and F1 score of 97%, as well as a recall of 96% on the test dataset. \
             The high precision indicates that the model is predicting wildfires accurately, while the high recall underlines that the model rarely labels \
             images at risk of a wildfire as non-wildfire photos. Obtaining a high recall score is especially valuable in wildfire detection, as false negatives \
             can have a detrimental impact to human life and the environment. Ideally, the number of false negatives would be lower than the number of false positives. \
             The F1 score further demonstrates a good balance between precision and recall, suggesting robust overall performance. ')
    
    st.markdown('#### Bayesian Performance ####')
    st.write('...')

with tab3:
    n_samples = 1000
    wildfire_lats, wildfire_longs = get_sample_coordinates(n_samples=n_samples)
    nowildfire_lats, nowildfire_longs = get_sample_coordinates(n_samples=n_samples, 
                                                               get_wildfire_set=False)
    dataset = np.ones(n_samples).tolist() + np.zeros(n_samples).tolist()

    st.markdown(f'### Locations of {n_samples} dataset training wildfire images ###')
    data = pd.DataFrame({
        'latitude': wildfire_lats + nowildfire_lats,
        'longitude': wildfire_longs + nowildfire_longs,
        'dataset': dataset,
        # 'color': ['red', 'green']
    })

    st.map(data)

    # TODO: Figure out how to make points on map clickable