import streamlit as st
import os

# --- STEP 1: PYTORCH SECURITY BYPASS ---
import torch
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

# --- STEP 2: STABLE IMPORTS ---
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# --- STEP 3: UI & MODEL ---
st.set_page_config(page_title="FractureAI", page_icon="🦴", layout="wide")

@st.cache_resource
def load_bone_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        return None
    return YOLO(model_path)

model = load_bone_model()

# --- STEP 4: MAIN APP ---
st.title("🏥 Bone & Joint Fracture Detection")
st.markdown("---")

if model is None:
    st.error("❌ 'best.pt' not found in GitHub!")
    st.stop()

uploaded_file = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    if st.button("🔍 Run Diagnostic Analysis"):
        img_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        results = model.predict(source=img_cv, conf=0.25)
        
        st.image(results[0].plot(), caption="Detection Results", use_container_width=True)
        
        if len(results[0].boxes) > 0:
            labels = [model.names[int(c)] for c in results[0].boxes.cls]
            st.success(f"Findings: {len(labels)}")
            st.table(pd.Series(labels).value_counts())
        else:
            st.info("No anomalies detected.")
