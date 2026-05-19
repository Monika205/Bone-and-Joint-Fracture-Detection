import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FractureAI",
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
    st.title("Diagnostic Center")

    st.success("🚀 Status: System Live")

    st.markdown("""
    **Developed By:**  
    ## Monika
    """)

    st.markdown("---")

    # ACCURACY CONTROL
    st.header("Sensitivity Control")

    conf_threshold = st.slider(
        "Confidence Threshold",
        0.0,
        1.0,
        0.50,
        0.05
    )

    st.info(
        "💡 Tip: Increasing this to 0.50+ will help ignore background text markers on the X-ray."
    )

# --- 4. MAIN INTERFACE ---
st.title("🏥 Bone & Joint Fracture Detection")
st.subheader("Professional CDSS")
st.markdown("---")

if model is None:
    st.error("❌ 'best.pt' not found. Please ensure the model file is in your GitHub.")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload Radiograph (X-ray)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Original Scan", use_container_width=True)

        analyze_btn = st.button("🔍 Run Full Diagnostic Analysis")

    if analyze_btn:

        with st.spinner('AI Associate Engineer System analyzing...'):

            img_cv = cv2.cvtColor(
                np.array(image.convert("RGB")),
                cv2.COLOR_RGB2BGR
            )

            # Run Prediction
            results = model.predict(
                source=img_cv,
                conf=conf_threshold
            )

            # LABEL FIX
            label_map = {
                "text": "Fracture Detected",
                "fracture": "Fracture Detected",
                "bone": "Skeletal Structure"
            }

            with col2:

                # Plot visual results
                res_plotted = results[0].plot()

                st.image(
                    res_plotted,
                    caption="AI Detection Results",
                    use_container_width=True
                )

                # RESULTS TABLE
                st.subheader("📊 Diagnostic Summary")

                boxes = results[0].boxes

                if len(boxes) > 0:

                    data = []

                    for box in boxes:

                        original_label = model.names[int(box.cls)]

                        display_label = label_map.get(
                            original_label,
                            "Anomaly Detected"
                        )

                        confidence = float(box.conf) * 100

                        data.append({
                            "Diagnosis": display_label,
                            "Accuracy Score": f"{confidence:.1f}%"
                        })

                    st.table(pd.DataFrame(data))

                    st.success(
                        f"✅ Analysis Complete: {len(boxes)} finding(s) identified."
                    )

                else:
                    st.info("✅ No fractures detected at the current threshold.")

# --- 5. PROFESSIONAL FOOTER ---
st.markdown("---")

footer = """
<div style="text-align: center; color: #6c757d; font-size: 14px;">
    &copy; 2026 Professional CDSS Project <br>
    Lead Engineer: <b>Monika</b>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)
