import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# --- STEP 1: UI SETUP ---
st.set_page_config(page_title="FractureAI", page_icon="🦴", layout="wide")

@st.cache_resource
def load_bone_model():
    if os.path.exists("best.pt"):
        return YOLO("best.pt")
    return None

model = load_bone_model()

# --- STEP 2: APP HEADER ---
st.title("🏥 Bone & Joint Fracture Detection")
st.write("Clinical Decision Support System (CDSS)")
st.markdown("---")

if model is None:
    st.error("❌ 'best.pt' not found in GitHub!")
    st.stop()

# --- STEP 3: SIDEBAR SETTINGS ---
st.sidebar.header("Settings")
conf_val = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# --- STEP 4: ANALYSIS ---
file = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png"])

if file:
    image = Image.open(file)
    st.image(image, caption="Original X-ray", width=400)
    
    if st.button("🔍 Run Full Analysis"):
        with st.spinner('Analyzing...'):
            img_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            results = model.predict(source=img_cv, conf=conf_val)
            
            # Show Detection
            st.image(results[0].plot(), caption="Detection Map", use_container_width=True)
            
            # Metrics Table
            boxes = results[0].boxes
            if len(boxes) > 0:
                st.success(f"Findings: {len(boxes)}")
                labels = [model.names[int(c)] for c in boxes.cls]
                st.table(pd.Series(labels).value_counts())
            else:
                st.info("No skeletal anomalies detected.")

st.sidebar.markdown("---")
st.sidebar.write("👤 **Developer:** Monika")
