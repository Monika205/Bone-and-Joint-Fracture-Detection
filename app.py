# --- UPDATED ANALYSIS BLOCK ---
if st.button("🔍 Run Full Diagnostic Analysis"):
    with st.spinner('Analyzing skeletal structure...'):
        img_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        
        # Run prediction with your slider value
        results = model.predict(source=img_cv, conf=conf_val)
        
        # FIX: Manually rename 'text' to 'Fracture Detected' for the display
        for result in results:
            for i, class_id in enumerate(result.boxes.cls):
                # If the model says 'text', we show 'Fracture'
                if model.names[int(class_id)] == "text":
                    model.names[int(class_id)] = "Fracture"

        res_plotted = results[0].plot()
        st.image(res_plotted, caption="Final Diagnostic Map", use_container_width=True)
        
        # Detailed Accuracy Table
        boxes = results[0].boxes
        if len(boxes) > 0:
            data = []
            for box in boxes:
                name = model.names[int(box.cls)]
                conf_score = float(box.conf) * 100
                data.append({"Finding": name, "Accuracy/Confidence": f"{conf_score:.2f}%"})
            
            st.table(pd.DataFrame(data))
