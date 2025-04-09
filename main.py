from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
import numpy as np
from PIL import Image
import io
import os

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Define model architecture exactly as in training
def create_model():
    input_tensor = Input(shape=(224, 224, 3))
    
    # Backbone: MobileNetV2 pretrained on ImageNet - IMPORTANT: use pretrained weights
    base_model = MobileNetV2(input_tensor=input_tensor, include_top=False, weights='imagenet')
    
    # Don't train the base model
    base_model.trainable = False
    
    # Head for 3-class classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(3, activation='softmax')(x)
    
    model = Model(inputs=input_tensor, outputs=output)
    
    # Compile the model as in training
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

# Create and load the model
try:
    # Create model with same architecture
    model = create_model()
    
    # Load weights
    model.load_weights('model_covid_classifier.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise e

# Define class labels (matching the exact order from training data generator)
CLASS_LABELS = ['Covid', 'Normal', 'Viral_Pneumonia']

def preprocess_image(image):
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Resize image to match model's expected sizing
    image = image.resize((224, 224))
    # Convert to numpy array and rescale exactly as in training
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Apply MobileNetV2 preprocessing
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array * 255.0)
    return img_array

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = CLASS_LABELS[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        
        return {
            "prediction": predicted_class,
            "confidence": confidence
        }
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return {
            "error": f"Error during prediction: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080) 