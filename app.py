import streamlit as st
import os

# --- STEP 1: SYSTEM OVERRIDE ---
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# --- STEP 2: PYTORCH SECURITY ---
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

# --- STEP 4: UI ---
st.set_page_config(page_title="FractureAI", page_icon="🦴")
st.title("🏥 Bone & Joint Fracture Detection")

@st.cache_resource
def get_model():
    if os.path.exists("best.pt"):
        return YOLO("best.pt")
    return None

model = get_model()

if model is None:
    st.error("❌ 'best.pt' missing from GitHub root folder.")
    st.stop()

# --- STEP 5: INFERENCE ---
file = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file)
    if st.button("🔍 Run Analysis"):
        # Convert PIL to BGR for YOLO
        img_cv = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        results = model.predict(source=img_cv, conf=0.25)
        
        # Plot and show
        st.image(results[0].plot(), caption="Detection Map", use_container_width=True)
        
        if len(results[0].boxes) > 0:
            st.success(f"Findings: {len(results[0].boxes)}")
        else:
            st.info("No fractures detected.")
