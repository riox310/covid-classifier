# COVID-19 Image Classification

A FastAPI web application for classifying chest X-ray images into three categories: COVID-19, Normal, and Viral Pneumonia.

## Features

- Upload chest X-ray images for classification
- Uses a pre-trained MobileNetV2 model
- Provides confidence scores for predictions
- Docker support for easy deployment
- Clean and modern web interface

## Prerequisites

- Python 3.9+
- Docker (optional)
- TensorFlow 2.12.0
- FastAPI
- Other dependencies listed in requirements.txt

## Installation

### Using Docker

1. Build the Docker image:
```bash
docker build -t covid-classifier .
```

2. Run the container:
```bash
docker run -p 5000:5000 -v ${PWD}/model_covid_classifier.h5:/app/model_covid_classifier.h5 covid-classifier
```

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/covid-classifier.git
cd covid-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## Usage

1. Access the web interface at `http://localhost:5000`
2. Upload a chest X-ray image
3. View the classification results and confidence score

## Project Structure

```
covid-classifier/
├── main.py              # FastAPI application
├── model.py             # Model loading and prediction
├── preprocessing.py     # Image preprocessing
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
├── .dockerignore       # Docker ignore file
├── static/             # Static files
├── templates/          # HTML templates
└── test_images/        # Test images directory
```

## Model Details

- Base Model: MobileNetV2
- Input Size: 224x224x3
- Classes: COVID-19, Normal, Viral Pneumonia
- Training: Transfer learning with ImageNet weights

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: COVID-19 Radiography Database
- Base Model: MobileNetV2
- Framework: FastAPI, TensorFlow 