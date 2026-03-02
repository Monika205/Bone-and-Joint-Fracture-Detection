import streamlit as st
import torch
import numpy as np
from PIL import Image
import pandas as pd
from fpdf import FPDF
from ultralytics import YOLO  # Standard YOLO class handles v10 automatically

# --- STEP 1: PYTORCH SECURITY OVERRIDE ---
import torch.serialization
torch.serialization.weights_only_default = False 

# --- STEP 2: CACHED MODEL LOADING ---
@st.cache_resource
def get_model():
    # 'best.pt' must be in the same folder as app.py on GitHub
    model = YOLO("best.pt") 
    return model

model = get_model()

# --- STEP 3: UI CONFIGURATION ---
st.set_page_config(page_title="FractureAI | Monika", page_icon="🏥", layout="wide")

st.title("🏥 Bone & Joint Fracture Detection (YOLOv10)")
st.write("Upload a Pediatric Wrist X-ray for AI-powered fracture analysis.")

# --- STEP 4: ANALYSIS LOGIC ---
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Original X-ray")
        st.image(image, use_container_width=True)

    conf_level = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)

    if st.button("🔍 Run Fracture Analysis"):
        with st.spinner("Analyzing bone structure..."):
            # Run prediction
            results = model.predict(source=image, conf=conf_level)
            res_plotted = results[0].plot() # This draws the boxes
            
            with col2:
                st.subheader("🛡️ AI Findings")
                st.image(res_plotted, use_container_width=True)
                
                detections = results[0].boxes
                if len(detections) > 0:
                    report_data = []
                    for box in detections:
                        report_data.append({
                            "Clinical Finding": model.names[int(box.cls[0])].upper(), 
                            "Confidence Score": f"{float(box.conf[0]):.2%}"
                        })
                    df = pd.DataFrame(report_data)
                    st.table(df)
                else:
                    st.success("✅ Analysis Complete: No significant anomalies detected.")

st.caption("© 2026 Monika | Pediatric Wrist Fracture Detection System")
