import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json # Import the json module to parse label_mapping.json
import io # Import io for BytesIO
import pandas as pd # Explicitly import pandas for debugging DataFrame creation

# --- Streamlit UI Layout (MUST BE FIRST Streamlit command in the script) ---
st.set_page_config(
    page_title="Visionary AI: Eye Disease Detection",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="auto"
)

# Define model and class names paths relative to the current script
MODEL_PATH = os.path.join('models', 'best_fundus_model.h5')
CLASS_NAMES_PATH = os.path.join('models', 'label_mapping.json')

# Image dimensions (must match training size)
IMG_HEIGHT, IMG_WIDTH = 224, 224

# --- Global Variables for Model and Class Names ---
@st.cache_resource
def load_model():
    """Loads the TensorFlow model."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Make sure '{MODEL_PATH}' exists and is a valid Keras model.")
        return None

@st.cache_data
def load_class_names():
    """
    Loads the class names from the JSON mapping file.
    Assumes label_mapping.json contains a dictionary like {"0": "class_A", "1": "class_B", ...}.
    """
    try:
        with open(CLASS_NAMES_PATH, 'r') as f:
            label_map = json.load(f)
        class_names = [label_map[str(i)] for i in range(len(label_map))]
        st.success("Class names loaded!")
        return class_names
    except FileNotFoundError:
        st.error(f"Error: Class names file not found at '{CLASS_NAMES_PATH}'.")
        return []
    except json.JSONDecodeError as e:
        st.error(f"Error parsing class names from JSON: {e}. Ensure '{CLASS_NAMES_PATH}' contains valid JSON.")
        return []
    except KeyError as e:
        st.error(f"Error: Missing key '{e}' in label mapping. Ensure all numerical keys (0 to N-1) are present.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while loading class names: {e}.")
        return []

# Call loading functions after set_page_config
model = load_model()
class_names = load_class_names()

# --- Preprocessing Function ---
def preprocess_image(image_bytes: bytes):
    """
    Preprocesses the uploaded image bytes for model inference.
    This must exactly match the preprocessing used during training (e.g., resizing, normalization).
    """
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    image_array = np.array(image)

    if image_array.ndim == 2:
        image_array = np.stack((image_array,)*3, axis=-1)
    elif image_array.ndim == 3 and image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array, image

st.title("👁️ Visionary AI: Hacking Blindness Before It Begins")
st.markdown("""
This application uses an AI model to detect potential eye diseases from fundus camera images.
Upload an image and get an instant prediction!
""")

# File uploader widget
uploaded_file = st.file_uploader("Choose a fundus image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.subheader("Uploaded Image:")
    st.image(uploaded_file, caption='Image for analysis', use_column_width=True)

    if model is not None and class_names:
        with st.spinner('Analyzing image... Please wait.'):
            processed_image_array, _ = preprocess_image(uploaded_file.read())
            predictions = model.predict(processed_image_array)

            predicted_probabilities = predictions[0]
            predicted_class_index = np.argmax(predicted_probabilities)

            if 0 <= predicted_class_index < len(class_names):
                predicted_class_name = class_names[predicted_class_index]
                confidence = float(predicted_probabilities[predicted_class_index])

                st.success("Analysis Complete!")
                st.subheader("Prediction Result:")
                st.write(f"**Predicted Disease:** <span style='color:#007bff; font-size:1.2em;'>{predicted_class_name}</span>", unsafe_allow_html=True)
                st.write(f"**Confidence:** <span style='color:#007bff; font-size:1.2em;'>{(confidence * 100):.2f}%</span>", unsafe_allow_html=True)

                st.markdown("---")
                st.subheader("All Probabilities:")

                # --- DEBUGGING ADDITIONS START HERE ---
                st.write(f"Debug: type(predicted_probabilities): {type(predicted_probabilities)}")
                st.write(f"Debug: predicted_probabilities content (first 5): {predicted_probabilities[:5] if isinstance(predicted_probabilities, np.ndarray) else predicted_probabilities}")
                st.write(f"Debug: len(predicted_probabilities): {len(predicted_probabilities) if hasattr(predicted_probabilities, '__len__') else 'N/A'}")
                st.write(f"Debug: type(class_names): {type(class_names)}")
                st.write(f"Debug: class_names content (first 5): {class_names[:5]}")
                st.write(f"Debug: len(class_names): {len(class_names)}")
                # --- DEBUGGING ADDITIONS END HERE ---

                if len(predicted_probabilities) == len(class_names):
                    try:
                        # Attempt to create DataFrame to catch errors here
                        prob_data_dict = {
                            "Disease": class_names,
                            "Probability": [f"{(p * 100):.2f}%" for p in predicted_probabilities]
                        }
                        # Convert to Pandas DataFrame explicitly before passing to st.dataframe
                        prob_data = pd.DataFrame(prob_data_dict)
                        st.write(f"Debug: Successfully created pandas DataFrame. Type: {type(prob_data)}")
                        st.write(f"Debug: First 5 rows of prob_data DataFrame:\n{prob_data.head()}")

                        st.dataframe(prob_data, hide_index=True) # Line 147 in original code, adjust after adding debugs
                    except Exception as e:
                        st.error(f"Error creating or displaying probability DataFrame: {e}")
                        st.write("Debug: Raw data for DataFrame creation:")
                        st.json({"Disease": class_names, "Probability": [str(p) for p in predicted_probabilities]}) # Display raw data as JSON
                else:
                    st.warning("Mismatch between number of predicted probabilities and class names. Cannot display all probabilities correctly.")
                    st.write("Predicted probabilities (raw):", predicted_probabilities)
                    st.write("Loaded class names:", class_names)

            else:
                st.error("Prediction result index out of bounds for loaded class names. This might indicate an issue with the model's output or class name loading.")
                st.write(f"Predicted index: {predicted_class_index}, Number of class names: {len(class_names)}")

    elif model is None:
        st.warning("Model could not be loaded. Please check the model path and file.")
    elif not class_names:
        st.warning("Class names could not be loaded or are empty. Please check the class names file content.")

st.markdown("---")
st.markdown("""
<p style='font-size:0.8em; color:#777;'>
    Disclaimer: This tool is for demonstration purposes only and should not be used for medical diagnosis.
    Always consult a qualified medical professional for any health concerns.
</p>
""", unsafe_allow_html=True)
