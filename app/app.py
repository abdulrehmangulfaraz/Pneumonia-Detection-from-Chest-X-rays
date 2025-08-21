import streamlit as st
import cv2
import numpy as np
import joblib
import pandas as pd
from PIL import Image
import os
from tensorflow.keras.models import load_model
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Pneumonia AI Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a Polished Dark Theme ---
def load_css():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
            
            html, body, [class*="st-"] {
                font-family: 'Inter', sans-serif;
            }

            /* Main container styling */
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            
            /* Custom Card Styling with Animation */
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .custom-card {
                background: rgba(40, 42, 54, 0.6);
                border-radius: 15px;
                padding: 25px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(12px);
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
                animation: fadeIn 0.5s ease-out;
            }
            
            /* Metric styling */
            .stMetric {
                background-color: rgba(0,0,0,0.3);
                border-radius: 10px;
                padding: 15px;
                text-align: center;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }

        </style>
    """, unsafe_allow_html=True)

load_css()

# --- Model Loading and Preprocessing (Cached for performance) ---
@st.cache_resource
def load_all_models():
    models = {}
    models_dir = 'models/'
    model_files = { "Logistic Regression": "Logistic_Regression.joblib", "KNN k=5": "KNN_k=5.joblib", "Naive Bayes": "Naive_Bayes.joblib", "SVM": "SVM.joblib", "Random Forest": "Random_Forest.joblib", "CNN": "CNN.keras" }
    for name, filename in model_files.items():
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            if filename.endswith(".joblib"): models[name] = joblib.load(path)
            elif filename.endswith(".keras"): models[name] = load_model(path)
    return models

models = load_all_models()

def preprocess_for_sklearn(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY) if len(np.array(image).shape) > 2 else np.array(image)
    resized_image = cv2.resize(gray_image, (64, 64))
    return resized_image.flatten().reshape(1, -1)

def preprocess_for_cnn(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY) if len(np.array(image).shape) > 2 else np.array(image)
    resized_image = cv2.resize(gray_image, (64, 64))
    normalized_image = resized_image / 255.0
    return normalized_image.reshape(1, 64, 64, 1)

# --- Sidebar Controls ---
with st.sidebar:
    st.title("🎛️ Control Panel")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload a chest X-ray...", type=["jpeg", "jpg", "png"], label_visibility="collapsed")
    st.info("Upload an image to run a comparative analysis across all trained AI models.")
    st.markdown("---")
    st.warning("Disclaimer: This tool is for educational purposes and is not a substitute for professional medical advice.")

# --- Main Page ---
st.title("🤖 AI Diagnostic Dashboard for Pneumonia")

if uploaded_file is None:
    st.info("⬅️ **Welcome!** Please upload a patient's X-ray using the control panel on the left to begin the AI-powered analysis.", icon="👋")

else:
    image = Image.open(uploaded_file)
    results = []
    
    # Run diagnostics with a progress bar
    progress_bar = st.progress(0, text="Initializing analysis...")
    for i, (name, model) in enumerate(models.items()):
        time.sleep(0.1) # Simulate processing time
        progress_bar.progress((i + 1) / len(models), text=f"Analyzing with {name}...")
        
        if name == "CNN":
            processed_image = preprocess_for_cnn(image)
            prediction_proba = model.predict(processed_image, verbose=0)[0][0]
        else: # Scikit-learn models
            processed_image = preprocess_for_sklearn(image)
            prediction_proba = model.predict_proba(processed_image)[0][1]
        
        results.append({"Model": name, "Prediction": "Pneumonia" if prediction_proba > 0.5 else "Normal", "Confidence": prediction_proba if prediction_proba > 0.5 else 1 - prediction_proba})
    progress_bar.empty()

    results_df = pd.DataFrame(results)
    
    # --- Create a single card for all results ---
    with st.container():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        # --- Create two main columns for the dashboard layout ---
        col1, col2 = st.columns([2, 3])

        with col1:
            st.subheader("🖼️ Patient X-ray")
            st.image(image, use_container_width=True)
            
            st.subheader("🎯 Overall Consensus")
            pneumonia_votes = results_df[results_df["Prediction"] == "Pneumonia"].shape[0]
            st.metric(
                label="Majority Vote",
                value="Pneumonia" if pneumonia_votes > len(models) / 2 else "Normal",
                delta=f"{pneumonia_votes} out of {len(models)} models agree",
                delta_color="inverse" if pneumonia_votes > len(models) / 2 else "normal"
            )

        with col2:
            st.subheader("🔬 Detailed Analysis")
            most_confident_model = results_df.loc[results_df['Confidence'].idxmax()]
            st.metric(
                label="🏆 Most Confident Model",
                value=f"{most_confident_model['Prediction']} ({most_confident_model['Confidence']:.1%})",
                delta=most_confident_model['Model']
            )
            st.markdown("---")
            st.dataframe(results_df, use_container_width=True, column_config={"Model": "AI Model", "Prediction": "Diagnosis", "Confidence": st.column_config.ProgressColumn("Confidence Level", format="%.2f%%", min_value=0, max_value=1)}, hide_index=True)
        
        st.markdown('</div>', unsafe_allow_html=True)