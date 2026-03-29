import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FractureAI | Akoode Technology", 
    page_icon="🏥", 
    layout="wide"
)

# --- 2. LOAD AI MODEL ---
@st.cache_resource
def load_bone_model():
    if os.path.exists("best.pt"):
        return YOLO("best.pt")
    return None

model = load_bone_model()

# --- 3. SIDEBAR (Credits & Settings) ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hospital.png", width=80)
    st.title("Settings & Info")
    
    # Professional Internship Credit
    st.success("🚀 **Project Status: Live**")
    st.markdown("""
    **Developed By:**
    ## **Monika**
    *Intern AI Associate Engineer*
    **Akoode Technology**
    """)
    st.markdown("---")
    
    # Accuracy Controls
    st.header("Diagnostic Controls")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.45, 0.05)
    st.info("💡 Tip: Set above 0.50 to reduce background noise/text markers.")
    
    st.markdown("---")
    st.write("📍 **BMU - B.Tech Data Science**")

# --- 4. MAIN INTERFACE ---
st.title("🏥 Bone & Joint Fracture Detection")
st.subheader("Professional CDSS - Powered by Akoode Technology")
st.markdown("---")

if model is None:
    st.error("❌ Model file 'best.pt' not found in root directory.")
    st.stop()

# File Upload Section
uploaded_file = st.file_uploader("Upload Radiograph (X-ray)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Split screen into two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded Scan", use_container_width=True)
        analyze_btn = st.button("🔍 Run Full Diagnostic Analysis")

    if analyze_btn:
        with st.spinner('AI is analyzing skeletal structure...'):
            # Image Processing
            img_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            
            # Prediction
            results = model.predict(source=img_cv, conf=conf_threshold)
            
            # Rename "text" labels for the visual plot
            for result in results:
                if result.boxes is not None:
                    for i, class_id in enumerate(result.boxes.cls):
                        label = model.names[int(class_id)]
                        if label == "text":
                            model.names[int(class_id)] = "Fracture/Anomaly"

            with col2:
                # Plot results
                res_plotted = results[0].plot()
                st.image(res_plotted, caption="AI Detection Results", use_container_width=True)
                
                # --- RESULTS TABLE ---
                st.subheader("📊 Diagnostic Summary")
                boxes = results[0].boxes
                if len(boxes) > 0:
                    data = []
                    for box in boxes:
                        label = model.names[int(box.cls)]
                        confidence = float(box.conf) * 100
                        data.append({
                            "Type": label, 
                            "Confidence": f"{confidence:.1f}%"
                        })
                    
                    st.table(pd.DataFrame(data))
                    st.success(f"✅ Total Potential Findings: {len(boxes)}")
                else:
                    st.warning("No anomalies detected at this threshold.")

# --- 5. PROFESSIONAL FOOTER ---
st.markdown("---")
footer_html = """
<div style="text-align: center;">
    <p style="color: #6c757d; font-size: 14px;">
        © 2026 Professional CDSS Project | <b>Akoode Technology</b> <br>
        Developed by <b>Monika</b> (Intern AI Associate Engineer)
    </p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
