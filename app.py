import streamlit as st
import os

# --- STEP 1: PYTORCH SECURITY OVERRIDE (MUST BE FIRST) ---
# This prevents the "UnpicklingError" on the server
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
    try:
        import torch.serialization
        torch.serialization.weights_only_default = False
    except:
        pass

# --- STEP 2: UI CONFIGURATION ---
st.set_page_config(page_title="FractureAI | Bone & Joint", page_icon="🦴", layout="wide")

# Custom Styling
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

# --- STEP 3: CACHED IMPORTS & MODEL LOADING ---
@st.cache_resource
def load_resources():
    # Import these inside the function to prevent line 4 crash during boot
    import cv2
    import numpy as np
    from PIL import Image
    import pandas as pd
    from ultralytics import YOLO
    
    model_path = "best.pt"
    if not os.path.exists(model_path):
        return None, None, None, None, None
        
    model = YOLO(model_path)
    return model, cv2, np, Image, pd

model, cv2, np, Image, pd = load_resources()

# --- STEP 4: APP HEADER ---
st.title("🏥 Bone & Joint Fracture Detection System")
st.subheader("Clinical Decision Support System (CDSS) powered by YOLOv10")
st.markdown("---")

if model is None:
    st.error("❌ Model 'best.pt' not found or failed to load. Please check your GitHub repo.")
    st.stop()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload Radiograph")
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original X-ray Image", use_container_width=True)

# --- STEP 5: INFERENCE & RESULTS ---
if uploaded_file is not None:
    with col1:
        analyze_btn = st.button("🔍 Run Diagnostic Analysis")

    if analyze_btn:
        with st.spinner('Analyzing bone structures...'):
            # Convert PIL to OpenCV format
            img_array = np.array(image.convert("RGB"))
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Run YOLOv10 Inference
            results = model.predict(source=img_cv, conf=0.25)
            res_plotted = results[0].plot()
            
            with col2:
                st.markdown("### 🎯 Detection Results")
                st.image(res_plotted, caption="Model Predictions", use_container_width=True)
                
                boxes = results[0].boxes
                if len(boxes) > 0:
                    st.success(f"✅ Findings Detected: {len(boxes)}")
                    detected_classes = [model.names[int(c)] for c in boxes.cls]
                    counts = pd.Series(detected_classes).value_counts()
                    st.write("**Analysis Summary:**")
                    st.dataframe(counts)
                else:
                    st.info("No fractures or significant bone anomalies detected.")

# --- STEP 6: SIDEBAR / CREDITS ---
st.sidebar.image("https://www.bml.edu.in/wp-content/uploads/2023/04/BML-Logo.png", width=150)
st.sidebar.title("System Information")
st.sidebar.markdown("---")
st.sidebar.write("👤 **Lead Developer:** Monika")
st.sidebar.write("🎓 **Institution:** BML Munjal University")
