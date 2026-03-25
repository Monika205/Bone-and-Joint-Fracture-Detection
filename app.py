import streamlit as st
import torch
import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from fpdf import FPDF

# --- STEP 1: PYTORCH SECURITY OVERRIDE ---
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

# --- STEP 3: CACHED MODEL LOADING ---
@st.cache_resource
def load_bone_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        return None
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"❌ Initialization Error: {e}")
        return None

model = load_bone_model()

# --- STEP 4: SIDEBAR CONTROLS (NEW) ---
st.sidebar.image("https://www.bml.edu.in/wp-content/uploads/2023/04/BML-Logo.png", width=150)
st.sidebar.title("🛠️ Analysis Settings")

# This slider lets you see "hidden" detections if the model is less than 25% sure
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.05, 
    max_value=1.0, 
    value=0.25, 
    help="Lower values show more (but less certain) detections. Higher values show only high-certainty detections."
)

st.sidebar.markdown("---")
st.sidebar.write("👤 **Lead Developer:** Monika")
st.sidebar.write("🎓 **Institution:** BML Munjal University")

# --- STEP 5: APP HEADER ---
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

# --- STEP 6: INFERENCE & DETAILED RESULTS ---
if uploaded_file is not None:
    with col1:
        analyze_btn = st.button("🔍 Run Diagnostic Analysis")

    if analyze_btn:
        if model:
            with st.spinner('Analyzing bone structures...'):
                img_array = np.array(image.convert("RGB"))
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                # Predict using the slider value
                results = model.predict(source=img_cv, conf=conf_threshold)
                res_plotted = results[0].plot()
                
                with col2:
                    st.markdown("### 🎯 Detection Results")
                    st.image(res_plotted, caption="Model Predictions", use_container_width=True)
                    
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        st.success(f"✅ Findings Detected: {len(boxes)}")
                        
                        # BUILD DETAILED DATA FRAME
                        data = []
                        for box in boxes:
                            cls_id = int(box.cls[0])
                            label = model.names[cls_id]
                            conf = float(box.conf[0])
                            data.append({"Finding": label, "Confidence (%)": f"{conf*100:.1f}%"})
                        
                        df_details = pd.DataFrame(data)
                        
                        st.write("**Detailed Diagnostic Report:**")
                        st.dataframe(df_details, use_container_width=True)
                        
                        # Warning if only 'text' is found
                        if all(df_details["Finding"] == "text"):
                            st.warning("Note: Only anatomical markers (text) detected. No fractures identified.")
                    else:
                        st.info("No fractures or significant bone anomalies detected at this confidence level.")
        else:
            st.error("Model is not loaded. Check 'best.pt' in your repo.")
