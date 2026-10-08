import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Pneumonia AI Diagnostic Suite",
    page_icon="🩺",
    layout="wide"
)

# --- Aesthetic Custom CSS Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
        background-color: #0284c7;
        color: white;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0369a1;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

IMG_HEIGHT, IMG_WIDTH = 150, 150

# --- Load Model Directly from .h5 ---
@st.cache_resource
def load_model_file():
    model_path = "chest_xray_model.h5"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital-3.png", width=64)
    st.title("MediScan AI")
    st.markdown("---")
    st.markdown("### System Status")
    model = load_model_file()
    if model is not None:
        st.success("🟢 Model Online (.h5 Loaded)")
    else:
        st.error("🔴 Model File Not Found")
    st.markdown("---")
    st.caption("Production Ready • v1.0")

# --- Main Dashboard ---
st.title("🩺 Clinical Imaging & Diagnostic Suite")
st.markdown("Upload a posterior-anterior chest X-ray digital radiograph for instant deep learning analysis.")
st.markdown("")

if model is None:
    st.warning("⚠️ `chest_xray_model.h5` was not found in the root directory. Please run your `model_training.py` script first.")
else:
    col_upload, col_results = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("### 📁 Input Radiograph")
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image file", 
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption='Loaded Patient Scan', use_column_width=True)

    with col_results:
        st.markdown("### 📊 Diagnostic Intelligence")
        
        if uploaded_file is not None:
            st.info("Radiograph staged and ready for inference.")
            
            if st.button("Execute CNN Screening Model", type="primary"):
                with st.spinner("Processing through convolutional layers..."):
                    # Preprocessing
                    img_resized = input_image.resize((IMG_WIDTH, IMG_HEIGHT))
                    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = img_array / 255.0  # Normalization
                    
                    # Prediction Inference
                    prediction = model.predict(img_array)[0][0]
                    
                    st.markdown("---")
                    st.subheader("Analysis Breakdown")
                    
                    if prediction > 0.5:
                        confidence = prediction * 100
                        st.error("### 🔴 Result: Pneumonia Positive")
                        st.markdown(f"**Confidence Score:** `{confidence:.2f}%`")
                        st.progress(int(confidence))
                        st.warning("⚠️ **Clinical Advisory:** Structural opacities detected. Immediate clinical correlation advised.")
                    else:
                        confidence = (1 - prediction) * 100
                        st.success("### 🟢 Result: Normal Scan")
                        st.markdown(f"**Confidence Score:** `{confidence:.2f}%`")
                        st.progress(int(confidence))
                        st.info("ℹ️ No prominent pathological indicators or fluid buildup detected within parameters.")
        else:
            st.markdown("""
                <div style="border: 2px dashed #e2e8f0; border-radius: 12px; padding: 40px; text-align: center; color: #94a3b8;">
                    <h4>Awaiting Scan Upload</h4>
                    <p>Please upload a chest X-ray file from the left panel to trigger the model pipeline.</p>
                </div>
            """, unsafe_allow_html=True)