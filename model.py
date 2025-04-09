import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load the model
def load_covid_model():
    try:
        model = load_model('model_covid_classifier.h5')
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise e

# Class labels
CLASS_LABELS = ['Covid', 'Normal', 'Viral_Pneumonia']

# Make prediction
def predict_image(model, processed_image):
    predictions = model.predict(processed_image)
    predicted_class = CLASS_LABELS[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    return predicted_class, confidence 