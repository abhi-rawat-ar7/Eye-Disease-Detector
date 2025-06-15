from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware # Required for frontend to communicate
import tensorflow as tf
import numpy as np
from PIL import Image # Pillow for image processing
import io
import os

# Define model and class names paths relative to the backend directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_fundus_model.h5')
CLASS_NAMES_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'label_mapping.json')

app = FastAPI()

# Configure CORS (Cross-Origin Resource Sharing)
# This is essential to allow your frontend (running on a different port/origin)
# to make requests to your backend.
origins = [
    "http://localhost",
    "http://localhost:8000", # Default FastAPI port
    "http://localhost:3000", # Common for React development server
    "http://127.0.0.1:5500", # Common for Live Server in VS Code
    # Add the URL of your deployed frontend if you deploy separately
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)

# Global variables to store the loaded model and class names
model = None
class_names = []
IMG_HEIGHT, IMG_WIDTH = 224, 224 # Must match training size

@app.on_event("startup")
async def load_model():
    """
    Load the TensorFlow model and class names when the FastAPI application starts up.
    This ensures the model is loaded only once and is available for all requests.
    """
    global model, class_names
    try:
        # Print the path to help with debugging FileNotFoundError
        print(f"Attempting to load model from: {MODEL_PATH}")
        # Load the Keras H5 model
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")

        # Load class names
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"Class names loaded: {class_names}")

    except Exception as e:
        raise RuntimeError(f"Could not load model or class names: {e}")

def preprocess_image(image_bytes: bytes):
    """
    Preprocesses the uploaded image bytes for model inference.
    This must exactly match the preprocessing used during training.
    """
    # Open image using Pillow
    image = Image.open(io.BytesIO(image_bytes))
    # Resize image to the target dimensions
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    # Convert to numpy array
    image_array = np.array(image)

    # Ensure image has 3 channels (RGB) - important if input is grayscale
    if image_array.ndim == 2: # Grayscale
        image_array = np.stack((image_array,)*3, axis=-1)
    elif image_array.ndim == 3 and image_array.shape[2] == 4: # RGBA to RGB
        image_array = image_array[:, :, :3]

    # Normalize pixel values to [0, 1]
    image_array = image_array / 255.0
    # Add batch dimension (model expects a batch of images)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.get("/")
async def read_root():
    """
    Root endpoint for testing API availability.
    """
    return {"message": "Welcome to the Eye Disease Detection API!"}

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """
    Endpoint to receive an image file, preprocess it, and return a prediction.
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded. Server is starting up or failed to load model.")

    try:
        # Read image file bytes
        image_bytes = await file.read()
        # Preprocess the image
        processed_image = preprocess_image(image_bytes)

        # Make prediction
        predictions = model.predict(processed_image)
        # Get the predicted class probabilities
        predicted_probabilities = predictions[0]
        # Get the index of the class with the highest probability
        predicted_class_index = np.argmax(predicted_probabilities)
        # Get the predicted class name
        predicted_class_name = class_names[predicted_class_index]
        # Get the confidence score for the predicted class
        confidence = float(predicted_probabilities[predicted_class_index])

        return JSONResponse(content={
            "filename": file.filename,
            "prediction": predicted_class_name,
            "confidence": f"{confidence:.4f}",
            "all_probabilities": {name: float(prob) for name, prob in zip(class_names, predicted_probabilities)}
        })

    except Exception as e:
        # Log the error for debugging
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
