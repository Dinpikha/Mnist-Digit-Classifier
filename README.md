

# MNIST Handwritten Digit Classification using CNN

This project is a Convolutional Neural Network (CNN) model built and trained to classify handwritten digits from the MNIST dataset. The model uses Keras/TensorFlow and achieves over 99% accuracy on both training and validation sets.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training Results](#training-results)

## Project Overview

The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), and the goal of this project is to build a model that can classify these digits with high accuracy. 

The project uses a Convolutional Neural Network (CNN) implemented in Keras and TensorFlow. The model achieves excellent results on the MNIST dataset, with a final validation accuracy of **98.83%** and training accuracy of **99.67%**.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/mnist-cnn-classification.git
    cd mnist-cnn-classification
    ```

2. **Install required dependencies:**
    Make sure you have Python 3.x installed, then install the necessary libraries using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

    Or, if you don't have `requirements.txt`, install individual dependencies:
    ```bash
    pip install tensorflow keras matplotlib numpy pandas seaborn
    ```

## Usage

1. **Run the model training:**

    ```bash
    python train_model.py
    ```

    This will load the MNIST dataset, preprocess it, train the CNN model, and save the trained model to a file (`mnist_model.keras`).

2. **Evaluate the model:**

    ```bash
    python evaluate_model.py
    ```

    This will load the saved model and display a plot of the training and validation accuracy over the epochs.

3. **Make Predictions:**

    Once the model is trained, you can load it and make predictions on new images:
    
    ```python
    from tensorflow.keras.models import load_model
    import numpy as np
    import matplotlib.pyplot as plt
    
    model = load_model('mnist_model.keras')
    # Assuming new_image is preprocessed and ready
    predictions = model.predict(new_image)  # Example: new_image should be reshaped (1, 28, 28, 1)
    predicted_label = np.argmax(predictions, axis=1)
    print("Predicted Label: ", predicted_label)
    ```

## Model Architecture

The model consists of the following layers:

1. **Input Layer**: 28x28 pixel grayscale images.
2. **Convolutional Layer**: 32 filters with a 3x3 kernel and ReLU activation.
3. **Max Pooling Layer**: 2x2 pool size.
4. **Convolutional Layer**: 64 filters with a 3x3 kernel and ReLU activation.
5. **Max Pooling Layer**: 2x2 pool size.
6. **Flatten Layer**: To convert the 2D feature maps into 1D.
7. **Fully Connected Layer**: 128 units with ReLU activation.
8. **Output Layer**: 10 units (for digits 0-9) with Softmax activation.

### The model is trained using:
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy

## Training Results

The model was trained for 10 epochs with the following performance:

- **Training Accuracy**: 
    - Epoch 1: 94.29%
    - Epoch 10: 99.67%
  
- **Validation Accuracy**:
    - Epoch 1: 97.65%
    - Epoch 10: 98.83%

### Training Accuracy (Pink) and Validation Accuracy (Orange):
View - Accuracy_graph --

---
