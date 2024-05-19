def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Enhance contrast using histogram equalization
    equalized_image = cv2.equalizeHist(blurred_image)
    
    # Detect edges using Canny edge detector
    edges = cv2.Canny(equalized_image, 100, 200)
    
    # Plot the original and processed images
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].imshow(original_image)
    axes[0, 1].set_title('Grayscale Image')
    axes[0, 1].imshow(gray_image, cmap='gray')
    axes[0, 2].set_title('Gaussian Blur')
    axes[0, 2].imshow(blurred_image, cmap='gray')
    axes[1, 0].set_title('Histogram Equalization')
    axes[1, 0].imshow(equalized_image, cmap='gray')
    axes[1, 1].set_title('Canny Edges')
    axes[1, 1].imshow(edges, cmap='gray')
    for ax in axes.flatten():
        ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Resize to a fixed size
    resized_image = cv2.resize(edges, (128, 128))
    
    # Convert processed image to 3-channel format
    processed_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
    
    # Normalize the image and convert to float32
    processed_image = processed_image.astype('float32') / 255.0
    
    return processed_image
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# Define your custom model class here
class CNNLSTM(tf.keras.models.Model):
    def __init__(self, num_filters, kernel_size, pool_size, lstm_units, dense_units, **kwargs):
        super(CNNLSTM, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, activation='relu')
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=pool_size)
        self.flatten = tf.keras.layers.Flatten()
        self.reshape = tf.keras.layers.Reshape(target_shape=(-1, num_filters))  # Reshape to (batch_size, timesteps, features)
        self.lstm = tf.keras.layers.LSTM(units=lstm_units)
        self.dense = tf.keras.layers.Dense(units=dense_units, activation='relu')
        self.output_layer = tf.keras.layers.Dense(units=3, activation='softmax')  # Assuming 3 classes: Abnormal, Covid, Normal

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.reshape(x)
        x = self.lstm(x)
        x = self.dense(x)
        return self.output_layer(x)

    def get_config(self):
        base_config = super(CNNLSTM, self).get_config()
        return {
            **base_config,
            "num_filters": self.conv.filters,
            "kernel_size": self.conv.kernel_size,
            "pool_size": self.pool.pool_size,
            "lstm_units": self.lstm.units,
            "dense_units": self.dense.units,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Enhance contrast using histogram equalization
    equalized_image = cv2.equalizeHist(blurred_image)
    
    # Detect edges using Canny edge detector
    edges = cv2.Canny(equalized_image, 100, 200)
    
    # Plot the original and processed images
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].imshow(original_image)
    axes[0, 1].set_title('Grayscale Image')
    axes[0, 1].imshow(gray_image, cmap='gray')
    axes[0, 2].set_title('Gaussian Blur')
    axes[0, 2].imshow(blurred_image, cmap='gray')
    axes[1, 0].set_title('Histogram Equalization')
    axes[1, 0].imshow(equalized_image, cmap='gray')
    axes[1, 1].set_title('Canny Edges')
    axes[1, 1].imshow(edges, cmap='gray')
    for ax in axes.flatten():
        ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Resize to a fixed size
    resized_image = cv2.resize(edges, (128, 128))
    
    # Convert processed image to 3-channel format
    processed_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
    
    # Normalize the image and convert to float32
    processed_image = processed_image.astype('float32') / 255.0
    
    return processed_image

def prepare_image_for_model(image_array):
    image_tensor = np.expand_dims(image_array, axis=0)
    return image_tensor

def load_model(model_path):
    custom_objects = {'CNNLSTM': CNNLSTM}
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

def predict(model, image_tensor):
    prediction = model.predict(image_tensor)
    return np.argmax(prediction, axis=1)[0]

# Streamlit app
st.title('ECG Image Classification')

# Upload image
uploaded_file = st.file_uploader("Choose an ECG image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_path = f"{uploaded_file.name}"
    image.save(image_path)
    
    # Process and display the image
    processed_image = preprocess_image(image_path)
    
    # Prepare image for model
    image_tensor = prepare_image_for_model(processed_image)
    
    # Load the model and make a prediction
    model = load_model('LSTM_model.h5')  # Ensure this is the correct path to your model
    prediction = predict(model, image_tensor)
    
    class_names = {0: 'Abnormal', 1: 'Covid', 2: 'Normal'}
    predicted_class = class_names[prediction]
    
    st.write(f'Predicted class: {predicted_class}')
