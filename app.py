import streamlit as st
import os

# --- STEP 1: FORCE SYSTEM TO USE HEADLESS MODE ---
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
except:
    import torch.serialization
    torch.serialization.weights_only_default = False

# --- STEP 3: IMPORTS ---
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# --- STEP 4: UI SETUP ---
st.set_page_config(page_title="FractureAI", page_icon="🦴")
st.title("🏥 Bone & Joint Fracture Detection")

@st.cache_resource
def load_model():
    if os.path.exists("best.pt"):
        return YOLO("best.pt")
    return None

model = load_model()

if model is None:
    st.error("❌ 'best.pt' file not found in your GitHub root!")
    st.stop()

# --- STEP 5: ANALYSIS ---
uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    if st.button("🔍 Run Full Analysis"):
        with st.spinner('Analyzing Radiograph...'):
            # Convert and Predict
            img_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            results = model.predict(source=img_cv, conf=0.25)
            
            # Display
            st.image(results[0].plot(), caption="Detection Map", use_container_width=True)
            
            if len(results[0].boxes) > 0:
                st.success(f"✅ Detection Successful: {len(results[0].boxes)} finding(s)")
            else:
                st.info("No skeletal anomalies detected.")
