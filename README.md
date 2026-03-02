🏥 Pediatric Bone & Joint Fracture Detection (YOLOv10)


📝 Project Overview

This project is an AI-driven Clinical Decision Support System (CDSS) designed for the automated detection and localization of fractures in pediatric wrist joints. Using the state-of-the-art YOLOv10 architecture, the system analyzes digital radiographs (X-rays) to assist radiologists in identifying subtle bone breaks, lesions, and anomalies in complex skeletal structures.

🦴 Key Clinical Focus: Bone & Joint

The system is specifically trained on the GRAZPEDWRI-DX dataset (20,000+ images), focusing on the pediatric wrist—one of the most challenging areas in orthopedics due to the presence of growth plates which often mimic fractures.

🚀 Technical Innovations

Architecture: Implemented YOLOv10, utilizing NMS-Free training (Non-Maximum Suppression) to eliminate redundant boxes and improve detection clarity in overlapping joint areas.

Dual Label Assignment: Uses one-to-many and one-to-one label assignments during training to ensure high precision in identifying small hairline fractures.

Medical Augmentation: Specialized preprocessing including Contrast Enhancement and Sharpening to handle low-quality or noisy clinical images.

📊 Performance & Findings
The model is optimized to detect 9 specific clinical categories:

Fracture (Bone breaks)

Periosteal Reaction (New bone growth/healing)

Bone Anomaly (Structural irregularities)

Bone Lesion

Foreign Body

Soft Tissue Swelling

...and more.

Benchmark Results:

Accuracy (mAP@50): ~51.9% (State-of-the-Art for this dataset)

Inference Speed: < 20ms (Real-time capability for ER settings)

💻 How to Run Locally
Clone the Repo:

Bash
git clone https://github.com/Monika205/Bone-and-Joint-Fracture-Detection.git
cd Bone-and-Joint-Fracture-Detection
Install Dependencies:

Bash
pip install -r requirements.txt
Run the App:

Bash
streamlit run app.py
📁 Repository Structure
app.py: The main Streamlit web application.

best.pt: The trained YOLOv10 weights (Bone & Joint optimized).

args.yaml: Hyperparameters used during the deep training phase.

requirements.txt: Environment configuration for Cloud/Local deployment.

packages.txt: System-level dependencies for Linux/Streamlit Cloud.

👩‍🔬 Research & Development
Developed as a professional Bone and Joint diagnostic tool.

Developer: Monika(For Intern opportunity at Akoode Technology)

Institute: BML Munjal University

Technology Stack: PyTorch, Ultralytics, OpenCV, Streamlit.
