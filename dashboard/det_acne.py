import streamlit as st

def display():
    from keras.models import load_model
    from keras.preprocessing import image
    from keras.layers import Input, TFSMLayer
    from keras.models import Model
    from PIL import Image, ImageOps
    import numpy as np

    # Function to preprocess image for prediction
    def preprocess_image(img_path, target_size=(224, 224)):
        try:
            img = Image.open(img_path).convert("RGB")
            img = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)
            img_array = np.asarray(img)
            normalized_img = (img_array.astype(np.float32) / 127.5) - 1
            data = np.expand_dims(normalized_img, axis=0)
            return data, img
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None, None

    # Fungsi untuk prediksi tipe jerawat
    def predict_acne_type(img_path, acne_type_model, labels):
        try:
            data, _ = preprocess_image(img_path)
            prediction = acne_type_model.predict(data, verbose=0)

            if isinstance(prediction, dict):
                prediction = prediction['sequential_7']

            index = np.argmax(prediction[0])  # Asumsi prediksi batch pertama
            acne_type = labels[index]
            return acne_type
        except Exception as e:
            st.error(f"Error during acne type prediction: {e}")
            return None

    # Fungsi untuk rekomendasi pengobatan
    def acne_treatment(acne_type):
        treatments = {
            "White Comedo": """Whiteheads, or closed comedones, are small or flesh-colored spots or bumps.
            
            How to Treat Whiteheads:
            
            Topical Salicylic Acid:
              - Neutrogena Oil-Free Acne Wash or Cleansing Pads (e.g., Almay Clear Complexion)
              - Salicylic acid helps clear clogs and reduce excess oil.
            Retinoid Use:
              - Differin Gel (Adapalene) or The Ordinary Retinol 0.2% in Squalane
              - Retinoids encourage skin cell turnover, reducing whiteheads.
            Clay Masks:
              - Aztec Secret Indian Healing Clay: Clay masks absorb excess oil and deep-clean pores.
            Gentle Exfoliation:
              - Paula's Choice 2% BHA Liquid Exfoliant for dead skin cell removal.
            DIY Oatmeal Mask:
              - Mix ground oatmeal with water or honey to reduce inflammation around whiteheads.
            """,
            "Nodule": """Nodules are hard, inflamed lumps located deep within the skin. They are flesh-colored or red bumps that are deeper than the skin’s surface and may or may not have pus.

            Modern Medicine:
            Oral Medications:
            - Doxycycline:
              - Vibramycin (doxycycline hyclate): ~$10-20/month (prescription).
              - Monodox (doxycycline monohydrate): ~$15-25/month (prescription).
            - Isotretinoin:
              - Accutane: ~$200-300/month (prescription).
              - Amnesteem: ~$200-300/month (prescription).
            
            Traditional Medicine:
            - Green Tea:
              - Twinings Green Tea Bags: ~$5-10 for a box of 20 bags.
              - Bigelow Green Tea: ~$5-10 for a box of 20 bags.
            - Neem Oil:
              - Organic Neem Oil by NOW Solutions: ~$10-15 for a 4 oz bottle.
              - Neem Aura Naturals Neem Oil: ~$10-15 for a 2 oz bottle.
            
            Additional Lifestyle Suggestions:
            - Skincare Routine: Use a gentle, non-comedogenic cleanser and moisturizer.
            - Dietary Considerations: Focus on anti-inflammatory foods and reduce sugar and dairy intake.
            - Hydration: Drink plenty of water.
            - Regular Exercise: Engage in at least 30 minutes of moderate exercise most days.
            - Stress Management: Practice relaxation techniques.
            - Sun Protection: Use an oil-free, broad-spectrum sunscreen.
            - Avoid Touching the Face.
            - Regular Dermatologist Visits.
            """,
            "Pustule": """Pustules are larger, tender bumps with a defined circular center filled with whitish or yellowish pus. They typically look like much larger and more inflamed whiteheads.

            Modern Medicine:
            Topical Treatments:
            - Salicylic Acid:
              - Neutrogena Oil-Free Acne Wash: ~$7-12
              - CeraVe Renewing SA Cleanser: ~$12-15
            - Retinoids:
              - Differin Gel (Adapalene): ~$10-15
              - Retinol 0.5% in Squalane by The Ordinary: ~$10
            
            Traditional Medicine:
            - Honey:
              - Manuka Honey (like Comvita Manuka Honey): ~$10-20 for a small jar.
              - Raw Honey from local sources: ~$5-10 for a bottle.
            - Apple Cider Vinegar:
              - Bragg Organic Raw Apple Cider Vinegar: ~$5-10 for a 16 oz bottle.
              - Heinz Apple Cider Vinegar: ~$3-5 for a bottle.
            
            Additional Lifestyle Suggestions:
            - Skincare Routine: Use a gentle cleanser twice daily and follow with a non-comedogenic moisturizer.
            - Exfoliation: Use a chemical exfoliant like Paula's Choice 2% BHA Liquid Exfoliant: ~$30.
            - Clay Masks: Aztec Secret Indian Healing Clay: ~$10 for a jar.
            - Diet: Incorporate more anti-inflammatory foods and reduce processed sugars and dairy products.
            - Hydration: Aim to drink at least 8 glasses of water a day.
            - Stress Management: Engage in regular physical activity or mindfulness practices.
            - Sun Protection: Apply EltaMD UV Clear Broad-Spectrum SPF 46: ~$40.
            - Avoid Touching the Face.
            """,
            "Papule": """Papules are bumps under the skin’s surface that are less than 1 cm in diameter. They appear solid, tender, and raised with inflamed surrounding skin.

            Modern Medicine:
            Topical Treatments:
            - Benzoyl Peroxide:
            - Clearasil Ultra Rapid Action Treatment Cream: ~$10-15
            - Neutrogena On-the-Spot Acne Treatment (benzoyl peroxide 2.5%): ~$6-10
            - Clindamycin:
            - Clindagel (Clindamycin Phosphate topical gel): ~$15-25 (prescription)
            - Duac Gel (Benzoyl Peroxide/Clindamycin): ~$30-50 (prescription)

            Traditional Medicine:
            - Tea Tree Oil:
            - The Body Shop Tea Tree Oil: ~$10-15 for a 0.5 oz bottle.
            - NOW Solutions Tea Tree Oil: ~$7-12 for a 1 oz bottle.
            - Aloe Vera Gel:
            - Nature Republic Aloe Vera 92% Soothing Gel: ~$10 for a 300 ml tube.
            - Lily of the Desert Aloe Vera Gel: ~$6-8 for a 12 oz bottle.

            Additional Lifestyle Suggestions:
            - Follow a consistent skincare routine:
            - Cleanse your face twice daily with a gentle, non-comedogenic cleanser like Cetaphil Gentle Skin Cleanser: ~$10.
            - Apply a non-comedogenic moisturizer, such as Neutrogena Hydro Boost Water Gel: ~$15.
            - Use a clay mask weekly to absorb excess oil and impurities, such as Aztec Secret Indian Healing Clay: ~$10.
            - Diet: Focus on a balanced diet rich in fruits and vegetables. Consider reducing dairy and processed foods.
            - Stay hydrated: Drink plenty of water.
            - Stress management: Practice yoga, meditation, or regular exercise.
            - Sun protection: Use a broad-spectrum sunscreen like Neutrogena Clear Face Sunscreen SPF 30: ~$10-15.
            - Avoid touching your face or picking at papules.
            """,
            "Black Comedo": """Blackheads, or open comedones, are small, dark-colored spots that may appear as slightly raised bumps.

            How to Treat Blackheads:
            
            Salicylic Acid Exfoliation:
              - Salicylic acid helps clean out blackheads by clearing pore blockages.
            Charcoal or Clay Masks:
              - Aztec Secret Indian Healing Clay or charcoal masks work deeply to pull out oil and debris.
            Regular Chemical Exfoliation:
              - Paula's Choice 2% BHA Liquid Exfoliant to reduce clogged pores and oxidation.
            Baking Soda Scrub:
              - Combine baking soda with water to make a gentle scrub for opening blackheads (use sparingly).
            Non-Comedogenic Oil Massage:
              - Light oils, like jojoba oil, can be used for gentle facial massage to lift blackheads naturally.
            """,
        }
        return treatments.get(acne_type, "No specific treatment available.")

    # Function to predict acne severity
    def predict_acne_severity(img_path, acne_severity_model, labels):
        try:
            data, img = preprocess_image(img_path)
            prediction = acne_severity_model.predict(data, verbose=0)
            
            if isinstance(prediction, dict):
                prediction = prediction['sequential_11']
            
            index = np.argmax(prediction[0])  # Assuming prediction is a batch
            severity_name = labels[index]
            return severity_name
        except Exception as e:
            st.error(f"Error during acne severity prediction: {e}")
            return None

    st.title("Acne Prediction and Treatment Recommendation")
    st.sidebar.header("Upload an Image")

    # Upload an image
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image")
        
        # Step 1: Load face detection model
        try:
            st.subheader("Step 1: Checking for a face...")
            face_model = load_model('model/my_face_classifier_model (2).h5')  # Replace with the path to your face detection model
            face_data, _ = preprocess_image(uploaded_file, target_size=(128, 128))
            prediction = face_model.predict(face_data, verbose=0)
            if prediction[0][0] > 0.5:
                st.warning("This is not a face. Please upload a valid face image.")
                return
            else:
                st.success("Face detected successfully!")
        except Exception as e:
            st.error(f"Error loading or using the face detection model: {e}")
            return
        
        # Step 2: Predict acne presence
        try:
            st.subheader("Step 2: Checking for acne presence...")
            acne_presence_model_path = "model/acne_nonacne/model.savedmodel"  # Replace with your SavedModel path
            labels_acne_presence = ["Non Acne", "Acne"]
            acne_presence_layer = TFSMLayer(acne_presence_model_path, call_endpoint='serving_default')
            input_shape = (224, 224, 3)
            inputs = Input(shape=input_shape)
            outputs_presence = acne_presence_layer(inputs)
            acne_presence_model = Model(inputs=inputs, outputs=outputs_presence)

            acne_presence_data, _ = preprocess_image(uploaded_file)
            presence_prediction = acne_presence_model.predict(acne_presence_data, verbose=0)
            
            if isinstance(presence_prediction, dict):
                # Mengakses key pertama dari dictionary
                presence_prediction = presence_prediction[next(iter(presence_prediction.keys()))]
            
            acne_presence_label = labels_acne_presence[np.argmax(presence_prediction)]
            
            st.info(f"Acne Presence: {acne_presence_label}")

            if acne_presence_label == "Non Acne":
                st.success("No acne detected! Your skin looks great.")
                return
        except Exception as e:
            st.error(f"Error loading or using the acne presence model: {e}")
            return

        # Step 3: Prediksi tipe jerawat
        try:
            st.subheader("Step 3: Predicting acne type...")
            acne_type_model_path = "model/acne_type/model.savedmodel"
            labels_acne_type = ["White Comedo", "Nodule", "Pustule", "Papule", "Black Comedo"]
            acne_type_layer = TFSMLayer(acne_type_model_path, call_endpoint='serving_default')
            outputs_type = acne_type_layer(inputs)
            acne_type_model = Model(inputs=inputs, outputs=outputs_type)

            acne_type = predict_acne_type(uploaded_file, acne_type_model, labels_acne_type)
            if acne_type:
                st.success(f"Predicted Acne Type: {acne_type}")

                # Step 4: Rekomendasi pengobatan
                st.subheader("Step 4: Treatment Recommendation")
                treatment = acne_treatment(acne_type)
                st.info(f"Recommended Treatment: {treatment}")
            else:
                st.error("Failed to determine acne type.")
        except Exception as e:
            st.error(f"Error during acne type prediction: {e}")
            return

        # Step 4: Predict acne severity
        try:
            st.subheader("Step 5: Predicting acne severity...")
            acne_severity_model_path = "model/acne_severity/model.savedmodel"  # Replace with your SavedModel path
            labels_acne_severity = ["Mild", "Moderate", "Severe", "Very Severe"]
            acne_severity_layer = TFSMLayer(acne_severity_model_path, call_endpoint='serving_default')
            outputs_severity = acne_severity_layer(inputs)
            acne_severity_model = Model(inputs=inputs, outputs=outputs_severity)

            severity_name = predict_acne_severity(uploaded_file, acne_severity_model, labels_acne_severity)
            if severity_name:
                st.success(f"Predicted Acne Severity: {severity_name}")
        except Exception as e:
            st.error(f"Error loading or using the acne severity model: {e}")