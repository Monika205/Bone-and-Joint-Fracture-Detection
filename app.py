import streamlit as st
import os

# --- STEP 1: FORCE HEADLESS MODE ---
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# --- STEP 2: PYTORCH SECURITY BYPASS ---
import torch
try:
    from torch.serialization import add_safe_globals
    add_safe_globals([
        'ultralytics.nn.tasks.DetectionModel', 'ultralytics.nn.modules.block.C2f',
        'ultralytics.nn.modules.conv.Conv', 'ultralytics.nn.modules.head.Detect',
        'ultralytics.nn.modules.block.DFL', 'ultralytics.nn.modules.block.SPPF',
        'ultralytics.nn.modules.conv.Concat'
    ])
except Exception:
    import torch.serialization
    torch.serialization.weights_only_default = False

# --- STEP 3: CORE IMPORTS ---
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# --- STEP 4: UI SETUP ---
st.set_page_config(page_title="FractureAI", page_icon="🦴", layout="wide")

@st.cache_resource
def load_model():
    if not os.path.exists("best.pt"):
        return None
    return YOLO("best.pt")

model = load_model()

st.title("🏥 Bone & Joint Fracture Detection")
st.markdown("---")

if model is None:
    st.error("❌ 'best.pt' not found in your repository.")
    st.stop()

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    if st.button("🔍 Run Analysis"):
        # Convert and Predict
        img_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        results = model.predict(source=img_cv, conf=0.25)
        
        # Display Results
        st.image(results[0].plot(), caption="Detection Map", use_container_width=True)
        
        if len(results[0].boxes) > 0:
            labels = [model.names[int(c)] for c in results[0].boxes.cls]
            st.success(f"Detected Findings: {len(labels)}")
            st.table(pd.Series(labels).value_counts())
        else:
            st.info("No anomalies detected.")

# --- SIDEBAR ---
st.sidebar.write(f"**Lead Developer:** Monika")
st.sidebar.write(f"**BMU Data Science**")
