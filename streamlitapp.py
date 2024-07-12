import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import streamlit.components.v1 as components

# Function to load data
def load_data():
    data = np.load('data_split.npz')
    X_test = data['X_test']
    y_test = data['y_test']
    return X_test, y_test

# Function to get Grad-CAM heatmap
def get_grad_cam(model, img_array, layer_name='conv2d_2'):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = gate_f * gate_r * grads
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam = np.zeros(output.shape[0:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
    cam = cv2.resize(cam.numpy(), (64, 64))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())
    return heatmap

# Load data and model
X_test, y_test = load_data()
model = load_model('models/malaria_model.h5')

# Ensure the model is built by calling it with some dummy data
dummy_data = np.zeros((1, 64, 64, 3))
model.predict(dummy_data)

# Streamlit app
st.title('Malaria Cell Classification')
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'png'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (64, 64))
    img_array = np.expand_dims(img, axis=0)
    prediction = model.predict(img_array)
    heatmap = get_grad_cam(model, img_array)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.imshow(heatmap, cmap='jet', alpha=0.5)
    st.pyplot(fig)
    st.write('Prediction: Parasitized' if prediction[0] > 0.5 else 'Prediction: Uninfected')

components.html(
    """
    <style>
    .sidebar .sidebar-content {
        width: 300px;
    }
    </style>
    """,
    height=0
)
