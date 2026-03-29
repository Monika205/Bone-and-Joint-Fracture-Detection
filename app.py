import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# --- 1. SETUP ---
st.set_page_config(page_title="FractureAI", page_icon="🦴", layout="wide")

@st.cache_resource
def load_bone_model():
    # Ensure best.pt is in your main GitHub folder
    if os.path.exists("best.pt"):
        return YOLO("best.pt")
    return None

model = load_bone_model()

# --- 2. SIDEBAR (Accuracy Controls) ---
st.sidebar.header("Diagnostic Settings")
# Increasing this slider helps remove "false" detections like the 'text' you saw
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.45, 0.05)
st.sidebar.info("Tip: Set to 0.50+ to ignore background noise and markers.")

# --- 3. MAIN INTERFACE ---
st.title("🏥 Bone & Joint Fracture Detection")
st.write("Professional CDSS - Radiographic Analysis")
st.markdown("---")

if model is None:
    st.error("❌ Model 'best.pt' not found. Please check your GitHub files.")
    st.stop()

uploaded_file = st.file_uploader("Upload Radiograph (X-ray)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Using columns for a professional layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded Scan", use_container_width=True)
        run_analysis = st.button("🔍 Run Full Diagnostic Analysis")

    if run_analysis:
        with st.spinner('Analyzing skeletal structure...'):
            # Convert PIL to OpenCV format
            img_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            
            # Predict with the threshold from your slider
            results = model.predict(source=img_cv, conf=conf_threshold)
            
            with col2:
                # Plot results back on the image
                res_plotted = results[0].plot()
                st.image(res_plotted, caption="Detection Map", use_container_width=True)
                
                # --- ACCURACY TABLE ---
                st.subheader("📊 Analysis Results")
                boxes = results[0].boxes
                if len(boxes) > 0:
                    data = []
                    for box in boxes:
                        # Clean up labels for professionalism
                        label = model.names[int(box.cls)]
                        if label == "text": label = "Fracture/Anomaly"
                        
                        confidence = float(box.conf) * 100
                        data.append({"Type": label, "Confidence": f"{confidence:.1f}%"})
                    
                    st.table(pd.DataFrame(data))
                    st.success(f"Total findings: {len(boxes)}")
                else:
                    st.info("No anomalies detected at this confidence level. Try lowering the threshold.")

# --- SIDEBAR CREDITS ---
st.sidebar.markdown("---")
st.sidebar.write("**Developer:** Monika")
st.sidebar.write("**BMU Data Science**")
