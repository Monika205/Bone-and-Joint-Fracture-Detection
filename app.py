import streamlit as st
import torch
import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from fpdf import FPDF

# --- STEP 1: PYTORCH SECURITY OVERRIDE (CRITICAL) ---
# This fixes the UnpicklingError in PyTorch 2.6+
try:
    from torch.serialization import add_safe_globals
    add_safe_globals([
        'ultralytics.nn.tasks.DetectionModel',
        'ultralytics.nn.modules.block.C2f',
        'ultralytics.nn.modules.conv.Conv',
        'ultralytics.nn.modules.head.Detect',
        'ultralytics.nn.modules.block.DFL',
        'ultralytics.nn.modules.block.SPPF',
        'ultralytics.nn.modules.conv.Concat'
    ])
except Exception:
    import torch.serialization
    torch.serialization.weights_only_default = False

from ultralytics import YOLO

# --- STEP 2: UI CONFIGURATION ---
st.set_page_config(page_title="FractureAI | Bone & Joint", page_icon="🦴", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        height: 3.5em; 
        background-color: #2E86C1; 
        color: white; 
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover { background-color: #21618C; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- STEP 3: CACHED MODEL LOADING ---
@st.cache_resource
def load_bone_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error(f"❌ Weights file '{model_path}' not found in the repository!")
        return None
    try:
        # Standard YOLO class handles v10 architecture automatically
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Initialization Error: {e}")
        return None

model = load_bone_model()

# --- STEP 4: APP HEADER ---
st.title("🏥 Bone & Joint Fracture Detection System")
st.subheader("Clinical Decision Support System (CDSS) powered by YOLOv10")
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload Radiograph")
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original X-ray Image", use_container_width=True)

# --- STEP 5: INFERENCE & RESULTS ---
if uploaded_file is not None:
    with col1:
        analyze_btn = st.button("🔍 Run Diagnostic Analysis")

    if analyze_btn:
        if model:
            with st.spinner('Analyzing bone structures...'):
                # Convert PIL to OpenCV format
                img_array = np.array(image.convert("RGB"))
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                # Run YOLOv10 Inference
                results = model.predict(source=img_cv, conf=0.25)
                res_plotted = results[0].plot()
                
                with col2:
                    st.markdown("### 🎯 Detection Results")
                    st.image(res_plotted, caption="Model Predictions", use_container_width=True)
                    
                    # Result Summary Logic
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        st.success(f"Findings Detected: {len(boxes)}")
                        
                        # Display specific findings
                        detected_classes = [model.names[int(c)] for c in boxes.cls]
                        counts = pd.Series(detected_classes).value_counts()
                        st.write("**Analysis Summary:**")
                        st.dataframe(counts, column_config={"index": "Finding", "0": "Count"})
                    else:
                        st.info("No fractures or significant bone anomalies detected.")
        else:
            st.error("Model is not loaded. Please check the 'best.pt' file.")

# --- STEP 6: SIDEBAR / CREDITS ---
st.sidebar.image("https://www.bml.edu.in/wp-content/uploads/2023/04/BML-Logo.png", width=150) # Optional logo placeholder
st.sidebar.title("System Information")
st.sidebar.info("""
This AI system is designed to assist radiologists in identifying skeletal fractures and joint anomalies with high precision using Deep Learning.
""")

st.sidebar.markdown("---")
st.sidebar.write("👤 **Lead Developer:** Monika")
st.sidebar.write("🎓 **Institution:** BML Munjal University")
st.sidebar.write("🔬 **Tech Stack:** YOLOv10, PyTorch, Streamlit")
