import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageStat
import pandas as pd
from fpdf import FPDF
from ultralytics import YOLO

# --- STEP 1: PYTORCH 2.6 SECURITY OVERRIDE ---
import torch.serialization
torch.serialization.weights_only_default = False 
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

# --- STEP 2: CACHED MODEL LOADING ---
@st.cache_resource
def get_model():
    return YOLO("best.pt")

model = get_model()

# --- STEP 3: HIGH-CONTRAST CLINICAL UI ---
st.set_page_config(page_title="FractureAI | Monika", page_icon="🏥", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0E1117; color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    [data-testid="stSidebar"] .stMarkdown p, h1, h2, h3 { color: #FFFFFF !important; }
    
    /* Force white background for tables for readability */
    .stTable, div[data-testid="stTable"] { 
        background-color: #FFFFFF !important; 
        border-radius: 8px; 
        padding: 10px;
    }
    .stTable td, .stTable th { color: #000000 !important; }

    .stButton>button { 
        width: 100%; border-radius: 8px; height: 3.5em; 
        background-color: #007BFF; color: white; font-weight: bold; border: none; 
    }
    </style>
    """, unsafe_allow_html=True)

# --- PDF REPORT ENGINE ---
def generate_clinical_report(role, q_score, findings_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(200, 15, txt="FractureAI Clinical Diagnostic Report", ln=True, align='C')
    
    pdf.set_font("Arial", 'B', 12)
    pdf.ln(10)
    pdf.cell(200, 10, txt="--- DEVELOPER INFORMATION ---", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 8, txt="Lead Developer: Monika (B.Tech Data Science)", ln=True)
    pdf.cell(200, 8, txt="Institution: BML Munjal University", ln=True)
    pdf.cell(200, 8, txt="Email: monikadhingra205@gmail.com", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="--- DIAGNOSTIC SUMMARY ---", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 8, txt=f"Authorized Access Level: {role}", ln=True)
    pdf.cell(200, 8, txt=f"Automated Image Quality Score: {q_score}/100", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="--- AI CLINICAL FINDINGS ---", ln=True)
    pdf.set_font("Arial", size=11)
    for _, row in findings_df.iterrows():
        pdf.cell(200, 8, txt=f"- {row['Clinical Finding']}: {row['Confidence Score']} Confidence", ln=True)
    
    pdf.ln(15)
    pdf.set_font("Arial", 'I', 9)
    pdf.multi_cell(0, 5, txt="DISCLAIMER: This system is a decision-support tool. All AI insights must be verified by a medical professional before clinical action.")
    return pdf.output(dest='S').encode('latin-1')

# --- SIDEBAR: AUTHOR & ROLES ---
st.sidebar.markdown("## 👩‍⚕️ Clinical Suite")
user_role = st.sidebar.selectbox("User Role", ["Healthcare Professional", "Patient"])

st.sidebar.markdown("---")
st.sidebar.subheader("Developer Profile")
st.sidebar.markdown("### **Monika**")
st.sidebar.markdown("🎓 B.Tech Data Science\n\n🏢 BML Munjal University")
st.sidebar.markdown("🔗 [LinkedIn Profile](https://www.linkedin.com/in/monika-dhingra-742b95304)")

st.sidebar.markdown("---")
conf_level = st.sidebar.slider("AI Sensitivity", 0.1, 1.0, 0.25)

# --- MAIN INTERFACE ---
st.title("🏥 FractureAI: Clinical Radiographic Suite")
st.markdown("---")

uploaded_file = st.file_uploader("📂 Upload Radiograph (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Image Quality Assessment (Section 3.2)
    brightness = ImageStat.Stat(image).mean[0]
    q_score = round(max(0, min(100, 100 - abs(128 - brightness))), 1)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📸 Radiograph Scan")
        st.image(image, use_container_width=True)
        st.metric("Image Quality Score", f"{q_score}/100")
        if q_score < 50:
            st.error("⚠️ Data Deficiency: Low quality image detected [Section 3.2].")

    if st.button("🔍 EXECUTE CLINICAL DIAGNOSTIC"):
        with st.spinner("AI analyzing bone structure..."):
            results = model.predict(source=np.array(image), conf=conf_level)
            res_plotted = results[0].plot()
            
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
                    st.table(df) # Force visible table
                    
                    # Generate and Download PDF Report (Section 3.6)
                    pdf_bytes = generate_clinical_report(user_role, q_score, df)
                    st.download_button(
                        label="📥 Download Detailed Clinical Report (PDF)",
                        data=pdf_bytes,
                        file_name="FractureAI_Clinical_Report.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.success("✅ Analysis Complete: No significant anomalies detected.")

st.markdown("---")
st.caption("© 2026 Monika | BML Munjal University | Professional CDSS Project for Akoode Technology")
