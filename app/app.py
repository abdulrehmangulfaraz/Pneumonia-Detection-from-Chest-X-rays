import streamlit as st
import cv2
import numpy as np
import joblib
import pandas as pd
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Pneumonia AI Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
def load_css():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
            html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
            .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
            .custom-card {
                background: rgba(40, 42, 54, 0.6); border-radius: 15px; padding: 25px;
                border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(12px);
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2); animation: fadeIn 0.5s ease-out;
            }
            .stMetric {
                background-color: rgba(0,0,0,0.3); border-radius: 10px; padding: 15px;
                text-align: center; border: 1px solid rgba(255, 255, 255, 0.1);
            }
        </style>
    """, unsafe_allow_html=True)

load_css()

# --- Model Loading (Cached) ---
@st.cache_resource
def load_all_models():
    models = {}
    models_dir = 'models/'
    model_files = {
        "CNN": "CNN.keras", "Logistic Regression": "Logistic_Regression.joblib",
        "Random Forest": "Random_Forest.joblib", "SVM": "SVM.joblib",
        "KNN k=5": "KNN_k=5.joblib", "Naive Bayes": "Naive_Bayes.joblib"
    }
    for name, filename in model_files.items():
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            if filename.endswith(".joblib"): models[name] = joblib.load(path)
            elif filename.endswith(".keras"): models[name] = load_model(path)
    return models

models = load_all_models()

# --- Model Info & Explicit Ranking ---
model_info = {
    "CNN": {"Accuracy": 0.96, "Type": "Deep Learning (Neural Network)"},
    "Logistic Regression": {"Accuracy": 0.95, "Type": "Linear Model (Regression-based)"},
    "SVM": {"Accuracy": 0.93, "Type": "Linear Model (Max-Margin)"},
    "Random Forest": {"Accuracy": 0.92, "Type": "Ensemble (Decision Trees)"},
    "KNN k=5": {"Accuracy": 0.88, "Type": "Instance-based (Proximity)"},
    "Naive Bayes": {"Accuracy": 0.86, "Type": "Probabilistic (Bayes' Theorem)"}
}

# The single source of truth for conflict resolution.
MODEL_RANKING = ["CNN", "Logistic Regression", "SVM", "Random Forest", "KNN k=5", "Naive Bayes"]
BEST_MODEL_NAME = MODEL_RANKING[0]

# --- Helper Functions ---
def preprocess_image(image, size, mode):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY) if len(np.array(image).shape) > 2 else np.array(image)
    resized_image = cv2.resize(gray_image, (size, size))
    if mode == 'cnn':
        return (resized_image / 255.0).reshape(1, size, size, 1)
    else: # sklearn
        return resized_image.flatten().reshape(1, -1)

def get_winner(df):
    """
    Finds the most confident model, breaking ties using the official MODEL_RANKING.
    """
    max_confidence = df['Confidence'].max()
    tied_models_df = df[df['Confidence'] == max_confidence]
    
    # Iterate through the official ranking to find the winner among the tied models
    for model_name in MODEL_RANKING:
        winner = tied_models_df[tied_models_df['Model'] == model_name]
        if not winner.empty:
            return winner.iloc[0] # Return the first one found
    return None # Should not happen

# --- Sidebar ---
with st.sidebar:
    st.title("🎛️ Control Panel")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload a chest X-ray...", type=["jpeg", "jpg", "png"], label_visibility="collapsed")
    st.info("Upload an image for a comprehensive AI-driven analysis.")
    st.markdown("---")
    st.warning("Disclaimer: This tool is for educational purposes only.")

# --- Main Page ---
st.title("🤖 AI Diagnostic Dashboard for Pneumonia")

if uploaded_file is None:
    st.info("⬅️ **Welcome!** Please upload a patient's X-ray to begin.", icon="👋")
else:
    image = Image.open(uploaded_file)
    results = []
    
    progress_bar = st.progress(0, text="Initializing analysis...")
    for name in MODEL_RANKING: # Process models in their ranked order
        model = models[name]
        time.sleep(0.1)
        progress_bar.progress((len(results) + 1) / len(models), text=f"Analyzing with {name}...")
        
        if isinstance(model, tf.keras.Model):
            processed_image = preprocess_image(image, 64, 'cnn')
            prediction_proba = model.predict(processed_image, verbose=0)[0][0]
        else:
            processed_image = preprocess_image(image, 64, 'sklearn')
            prediction_proba = model.predict_proba(processed_image)[0][1]
        
        prediction = "Pneumonia" if prediction_proba > 0.5 else "Normal"
        confidence = prediction_proba if prediction == "Pneumonia" else 1 - prediction_proba
        results.append({
            "Model": name, "Prediction": prediction, "Confidence": confidence,
            "Accuracy": model_info[name]['Accuracy'], "Type": model_info[name]['Type']
        })
    progress_bar.empty()

    results_df = pd.DataFrame(results)
    # Set the order for display based on the official ranking
    display_df = results_df.set_index('Model').loc[MODEL_RANKING].reset_index()

    with st.container():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 3])

        with col1:
            st.subheader("🖼️ Patient X-ray")
            st.image(image, use_container_width=True)
            
            st.subheader("🎯 Overall Consensus")
            
            model_weights = {k: v['Accuracy'] for k, v in model_info.items()}
            pneumonia_weight = sum(model_weights.get(row['Model'], 0) for _, row in results_df.iterrows() if row['Prediction'] == 'Pneumonia')
            total_weight = sum(model_weights.values())
            weighted_prediction = "Pneumonia" if pneumonia_weight > (total_weight / 2) else "Normal"

            final_prediction = weighted_prediction
            best_model_result = results_df[results_df['Model'] == BEST_MODEL_NAME].iloc[0]
            override_applied = False
            
            HIGH_CONFIDENCE_THRESHOLD = 0.98 
            if best_model_result['Confidence'] > HIGH_CONFIDENCE_THRESHOLD and best_model_result['Prediction'] != weighted_prediction:
                final_prediction = best_model_result['Prediction']
                override_applied = True

            final_confidence = results_df[results_df['Prediction'] == final_prediction]['Confidence'].mean() * 100
            
            st.metric(label="Final Verdict", value=final_prediction, delta=f"{final_confidence:.1f}% Confidence", delta_color="inverse" if final_prediction == "Pneumonia" else "normal")
            
            if override_applied:
                st.warning(f"Verdict overridden by high-confidence {BEST_MODEL_NAME} result.", icon="⚠️")

        with col2:
            st.subheader("🔬 Detailed Analysis")
            
            # Use the new guaranteed tie-breaking function
            winner = get_winner(results_df)
            
            st.metric(
                label="🏆 Most Confident Model",
                value=f"{winner['Prediction']} ({winner['Confidence']:.1%})",
                delta=winner['Model']
            )
            st.markdown("---")
            
            st.dataframe(
                display_df,
                use_container_width=True, hide_index=True,
                column_order=("Model", "Type", "Accuracy", "Prediction", "Confidence"),
                column_config={
                    "Model": "AI Model", "Type": "Model Type",
                    "Accuracy": st.column_config.NumberColumn("Training Accuracy", format="%.2f%%"),
                    "Prediction": "Diagnosis",
                    "Confidence": st.column_config.ProgressColumn("Confidence Level", format="%.2f", min_value=0, max_value=1),
                }
            )
        st.markdown('</div>', unsafe_allow_html=True)