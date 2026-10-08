# 🩺 Chest X-Ray Pneumonia Classification & Diagnostic Suite

A deep learning-powered medical imaging web application that leverages a custom **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras** to screen and classify chest X-ray radiographs for indicators of **Pneumonia**. The application features an aesthetic, production-ready interactive dashboard powered by **Streamlit**.

---

## 🚀 Features

* **Custom CNN Architecture:** Engineered with multiple convolutional and pooling blocks followed by dropout regularization to extract spatial features and prevent overfitting.
* **Streamlit Web Interface:** A modern, medical-grade UI featuring a split-layout design, live model status checks, interactive prediction progress bars, and custom clinical feedback cards.
* **Direct Model Persistence:** Trains and exports model weights cleanly into a standard `.h5` file format for fast and lightweight inference.

* 🧠 Model Architecture Overview
The underlying sequential network consists of:

Input Layer: Resizes incoming images to 150 x 150 pixels with 3 color channels (RGB).

Convolutional Blocks: 3 stacked Conv2D feature extraction blocks (32, 64, and 128 filters respectively) using ReLU activation paired with MaxPooling2D spatial downsampling.

Classifier Head: Flatten layer leading into a dense 128 neuron layer, a 0.5 Dropout layer for regularization, and a final sigmoid output node for binary classification (Normal vs. Pneumonia).

⚠️ Clinical Disclaimer
This software application is built strictly for educational, portfolio demonstration, and engineering research purposes. It must not be utilized as a substitute for professional clinical judgment, diagnosis, or treatment.
