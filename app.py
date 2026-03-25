import streamlit as st
import torch
import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from fpdf import FPDF

# --- STEP 1: PYTORCH SECURITY OVERRIDE (CRITICAL) ---
# This MUST happen before 'from ultralytics import YOLO'
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
st.set_page_config(page_title="FractureAI | Monika", page_icon="🦴", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; font-weight: bold; }
    .report-text { font-size: 18px; font-weight: bold; color: #333; }
    </style>
    """, unsafe_allow_html=True)

# --- STEP 3: CACHED MODEL LOADING ---
@st.cache_resource
def get_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error(f"❌ Error: '{model_path}' not found in the repository root!")
        return None
    try:
        # standard YOLO class handles v10 weights automatically
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Model Loading Failed: {e}")
        return None

model = get_model()

# --- STEP 4: APP HEADER ---
st.title("🏥 Pediatric Bone & Joint Fracture Detection")
st.markdown("### AI-Driven Clinical Decision Support System")
st.info("Upload a pediatric X-ray to detect fractures, lesions, or bone anomalies.")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload X-ray Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

# --- STEP 5: ANALYSIS & RESULTS ---
if uploaded_file is not None:
    if st.button("🔍 Run Fracture Analysis"):
        if model:
            # Prepare image for YOLO
            img_array = np.array(image.convert("RGB"))
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Run Inference
            results = model.predict(source=img_cv, conf=0.25)
            res_plotted = results[0].plot()
            
            with col2:
                st.image(res_plotted, caption="Analysis Result", use_container_width=True)
                
                # Summary Statistics
                boxes = results[0].boxes
                if len(boxes) > 0:
                    st.success(f"✅ Detection Complete: {len(boxes)} findings identified.")
                    
                    # Create a simple results table
                    detected_classes = [model.names[int(c)] for c in boxes.cls]
                    df_results = pd.DataFrame(detected_classes, columns=["Finding Type"])
                    st.table(df_results["Finding Type"].value_counts())
                else:
                    st.warning("⚠️ No fractures detected in this view.")
        else:
            st.error("Model initialization failed. Check logs.")

# --- STEP 6: FOOTER ---
st.sidebar.markdown("---")
st.sidebar.write("👤 **Developer:** Monika")
st.sidebar.write("🎓 **Institution:** BML Munjal University")
