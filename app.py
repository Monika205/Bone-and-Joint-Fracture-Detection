import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from fpdf import FPDF

# --- STEP 1: PYTORCH SECURITY OVERRIDE (CRITICAL) ---
# This must happen BEFORE loading the YOLO model to prevent UnpicklingError
try:
    from torch.serialization import add_safe_globals
    add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
except Exception:
    import torch.serialization
    torch.serialization.weights_only_default = False

from ultralytics import YOLO

# --- STEP 2: UI CONFIGURATION ---
st.set_page_config(page_title="FractureAI | Monika", page_icon="🦴", layout="wide")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_value=True)

# --- STEP 3: CACHED MODEL LOADING ---
@st.cache_resource
def get_model():
    # Ensure "best.pt" is in the same folder as this script on GitHub
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"Error: Could not load 'best.pt'. Ensure the file is in the repo. Details: {e}")
        return None

model = get_model()

# --- STEP 4: APP HEADER ---
st.title("🏥 Pediatric Bone & Joint Fracture Detection")
st.subheader("AI-Driven Clinical Decision Support System (YOLOv10)")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload X-ray Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    image = Image.open(uploaded_file)
    img_array = np.array(image.convert("RGB"))
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    if st.button("Analyze X-ray"):
        if model:
            # Run Inference
            results = model.predict(source=img_cv, conf=0.25)
            res_plotted = results[0].plot()
            
            with col2:
                st.image(res_plotted, caption="Analysis Result", use_container_width=True)
                
            # --- STEP 5: RESULTS SUMMARY ---
            boxes = results[0].boxes
            if len(boxes) > 0:
                st.success(f"Detection Complete: {len(boxes)} findings identified.")
                # Logic for report generation can go here
            else:
                st.info("No fractures or anomalies detected in this view.")
        else:
            st.error("Model not initialized.")

# --- STEP 6: FOOTER ---
st.sidebar.markdown("---")
st.sidebar.write("Developed by: **Monika**")
st.sidebar.write("Institution: **BML Munjal University**")
