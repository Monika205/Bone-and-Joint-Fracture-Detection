import streamlit as st
import torch
import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd

# --- STEP 1: PYTORCH SECURITY & YOLOv10 CONFIG ---
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
st.set_page_config(page_title="FractureAI | High Accuracy", page_icon="🦴", layout="wide")

# Custom Styling for Professional Look
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
        st.error(f"❌ Model Error: {e}")
        return None

model = load_bone_model()

# --- STEP 4: ACCURACY CONTROLS (SIDEBAR) ---
st.sidebar.image("https://www.bml.edu.in/wp-content/uploads/2023/04/BML-Logo.png", width=150)
st.sidebar.title("🛠️ Accuracy Settings")

# IMPORTANT: Adjust this slider if fractures aren't appearing
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    0.05, 1.0, 0.25, 
    help="Lower this if the model misses a fracture. Raise it to remove 'text' noise."
)

iou_threshold = st.sidebar.slider(
    "Overlapping (IOU) Threshold", 
    0.1, 1.0, 0.45, 
    help="Higher values allow boxes to overlap more (useful for complex fractures)."
)

st.sidebar.markdown("---")
st.sidebar.write("👤 **Lead Developer:** Monika")
st.sidebar.write("🎓 **Institution:** BML Munjal University")

# --- STEP 5: MAIN APP INTERFACE ---
st.title("🏥 Bone & Joint Fracture Detection System")
st.subheader("Advanced Diagnostic Analysis (YOLOv10)")
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload Radiograph")
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original X-ray Image", use_container_width=True)

# --- STEP 6: DIAGNOSTIC INFERENCE ---
if uploaded_file is not None:
    with col1:
        analyze_btn = st.button("🔍 Run Full Diagnostic Analysis")

    if analyze_btn:
        if model:
            with st.spinner('Analyzing bone integrity...'):
                img_array = np.array(image.convert("RGB"))
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                # Predict with custom thresholds for better accuracy
                results = model.predict(
                    source=img_cv, 
                    conf=conf_threshold, 
                    iou=iou_threshold
                )
                res_plotted = results[0].plot()
                
                with col2:
                    st.markdown("### 🎯 Detection Results")
                    st.image(res_plotted, caption="Diagnostic Map", use_container_width=True)
                    
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        st.success(f"✅ Findings Detected: {len(boxes)}")
                        
                        # BUILD ACCURATE RESULTS TABLE
                        report_data = []
                        for box in boxes:
                            label = model.names[int(box.cls[0])]
                            accuracy = float(box.conf[0])
                            report_data.append({
                                "Finding Type": label, 
                                "Accuracy (Confidence)": f"{accuracy*100:.2f}%"
                            })
                        
                        df_report = pd.DataFrame(report_data)
                        st.write("**Full Analysis Report:**")
                        st.dataframe(df_report, use_container_width=True)
                        
                        # Clinical Note if only text is found
                        if all(df_report["Finding Type"] == "text"):
                            st.warning("⚠️ Note: Only marker text found. No skeletal fractures identified.")
                    else:
                        st.info("No anomalies detected at this sensitivity level.")
        else:
            st.error("Weights file 'best.pt' missing or corrupted.")
