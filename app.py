import streamlit as st
import os

# --- STEP 1: PYTORCH SECURITY FIX (MUST BE FIRST) ---
try:
    import torch
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

# --- STEP 2: STABLE IMPORTS ---
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# --- STEP 3: UI CONFIGURATION ---
st.set_page_config(page_title="FractureAI | Bone & Joint", page_icon="🦴", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { 
        width: 100%; border-radius: 8px; height: 3.5em; 
        background-color: #2E86C1; color: white; font-weight: bold; border: none;
    }
    .stButton>button:hover { background-color: #21618C; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- STEP 4: MODEL LOADING ---
@st.cache_resource
def load_bone_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        return None
    return YOLO(model_path)

model = load_bone_model()

# --- STEP 5: MAIN INTERFACE ---
st.title("🏥 Bone & Joint Fracture Detection System")
st.subheader("Clinical Decision Support System (CDSS) - Powered by YOLOv10")
st.markdown("---")

if model is None:
    st.error("❌ 'best.pt' file not found. Please upload it to your GitHub root folder.")
    st.stop()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    uploaded_file = st.file_uploader("Upload Radiograph (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original X-ray", use_container_width=True)

# --- STEP 6: ACCURACY & DETECTION ---
if uploaded_file is not None:
    if st.button("🔍 Run Diagnostic Analysis"):
        with st.spinner('Analyzing skeletal integrity...'):
            img_array = np.array(image.convert("RGB"))
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Prediction with standard threshold for accuracy
            results = model.predict(source=img_cv, conf=0.25)
            res_plotted = results[0].plot()
            
            with col2:
                st.markdown("### 🎯 Detection Map")
                st.image(res_plotted, caption="Model Findings", use_container_width=True)
                
                boxes = results[0].boxes
                if len(boxes) > 0:
                    st.success(f"✅ Total Findings: {len(boxes)}")
                    labels = [model.names[int(c)] for c in boxes.cls]
                    st.write("**Analysis Summary Table:**")
                    st.table(pd.Series(labels).value_counts())
                else:
                    st.info("No skeletal anomalies detected at this confidence level.")

# --- STEP 7: SIDEBAR ---
st.sidebar.image("https://www.bml.edu.in/wp-content/uploads/2023/04/BML-Logo.png", width=150)
st.sidebar.markdown("---")
st.sidebar.write("👤 **Lead Developer:** Monika")
st.sidebar.write("🎓 **Institution:** BML Munjal University")
