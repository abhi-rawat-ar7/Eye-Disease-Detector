import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json # Import the json module to parse label_mapping.json
import io # Import io for BytesIO

# --- Streamlit UI Layout (MUST BE FIRST Streamlit command in the script) ---
# This is crucial to avoid StreamlitSetPageConfigMustBeFirstCommandError.
# Ensure no other `st.` calls (like st.title, st.markdown, st.sidebar, etc.)
# appear before st.set_page_config in your actual file.
st.set_page_config(
    page_title="Visionary AI: Eye Disease Detection",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="auto"
)

# Define model and class names paths relative to the current script
# Adjust these paths based on where you place your streamlit_app.py
# If streamlit_app.py is in the root, and models in a 'models' folder:
MODEL_PATH = os.path.join('models', 'best_fundus_model.h5') # Assuming you saved as best_fundus_model.h5
CLASS_NAMES_PATH = os.path.join('models', 'label_mapping.json') # Updated to point to label_mapping.json

# Image dimensions (must match training size)
IMG_HEIGHT, IMG_WIDTH = 224, 224

# --- Global Variables for Model and Class Names ---
# Use st.cache_resource to load the model only once
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

# Use st.cache_data to load class names only once
@st.cache_data
def load_class_names():
    """
    Loads the class names from the JSON mapping file.
    Assumes label_mapping.json contains a dictionary like {"0": "class_A", "1": "class_B", ...}.
    """
    try:
        with open(CLASS_NAMES_PATH, 'r') as f:
            label_map = json.load(f)
        # Convert dictionary values to a list, ensuring order by integer keys
        # This is crucial because model output indices correspond to these numerical keys.
        # Example: label_map = {"0": "Normal", "1": "DR1"} -> class_names = ["Normal", "DR1"]
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
    # Open image using Pillow from bytes
    image = Image.open(io.BytesIO(image_bytes))
    # Resize image to the target dimensions
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    # Convert to numpy array
    image_array = np.array(image)

    # Ensure image has 3 channels (RGB) - important if input is grayscale or RGBA
    if image_array.ndim == 2: # Grayscale
        image_array = np.stack((image_array,)*3, axis=-1) # Convert to 3 channels
    elif image_array.ndim == 3 and image_array.shape[2] == 4: # RGBA to RGB
        image_array = image_array[:, :, :3]

    # Normalize pixel values to [0, 1]
    image_array = image_array / 255.0
    # Add batch dimension (model expects a batch of images)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array, image # Return processed array and original PIL image for display


st.title("👁️ Visionary AI: Hacking Blindness Before It Begins")
st.markdown("""
This application uses an AI model to detect potential eye diseases from fundus camera images.
Upload an image and get an instant prediction!
""")

# File uploader widget
uploaded_file = st.file_uploader("Choose a fundus image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.subheader("Uploaded Image:")
    # We pass the file object directly to st.image, it handles reading bytes
    st.image(uploaded_file, caption='Image for analysis', use_column_width=True)

    # Perform prediction when model and class names are loaded
    if model is not None and class_names:
        with st.spinner('Analyzing image... Please wait.'):
            # Preprocess and predict
            # uploaded_file is a BytesIO-like object, use .read() to get raw bytes
            processed_image_array, _ = preprocess_image(uploaded_file.read())
            predictions = model.predict(processed_image_array)

            # Get the predicted class and confidence
            predicted_probabilities = predictions[0]
            predicted_class_index = np.argmax(predicted_probabilities)

            # Ensure the predicted index is within the bounds of class_names
            if 0 <= predicted_class_index < len(class_names):
                predicted_class_name = class_names[predicted_class_index]
                confidence = float(predicted_probabilities[predicted_class_index])

                st.success("Analysis Complete!")
                st.subheader("Prediction Result:")
                st.write(f"**Predicted Disease:** <span style='color:#007bff; font-size:1.2em;'>{predicted_class_name}</span>", unsafe_allow_html=True)
                st.write(f"**Confidence:** <span style='color:#007bff; font-size:1.2em;'>{(confidence * 100):.2f}%</span>", unsafe_allow_html=True)

                st.markdown("---")
                st.subheader("All Probabilities:")
                # Display all probabilities in a table or list
                # Ensure the length of probabilities matches class names
                if len(predicted_probabilities) == len(class_names):
                    prob_data = {
                        "Disease": class_names,
                        "Probability": [f"{(p * 100):.2f}%" for p in predicted_probabilities]
                    }
                    st.dataframe(prob_data, hide_index=True)
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