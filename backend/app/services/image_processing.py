
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

# Locate the trained V3 model
BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "ml"
    / "saved_models"
    / "mindease_emotion_model_v3.keras"
)

CLASS_NAMES = [
    "angry",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

IMAGE_SIZE = (224, 224)

# Load model once when the service is imported
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


def predict_emotion(image_bytes):
    """
    Predict facial expression from uploaded image bytes.
    """

    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image file.")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image
    image = cv2.resize(image, IMAGE_SIZE)

    # Normalize image to [0, 1]
    image = image.astype("float32") 

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Predict
    predictions = model.predict(image, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))

    return {
        "emotion": CLASS_NAMES[predicted_index],
        "confidence": round(
            float(predictions[predicted_index]) * 100, 2
        ),
        "probabilities": {
            emotion: round(float(score) * 100, 2)
            for emotion, score in zip(CLASS_NAMES, predictions)
        }
    }
