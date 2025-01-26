# model_experimentation_and_packaging

# ML Model Deployment Project

## Overview
Machine learning model with hyperparameter tuning and Docker deployment.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run hyperparameter tuning: `python src/hyperparameter_tuning.py`

## Deployment
1. Build Docker image: `docker build -t ml-model-service .`
2. Run container: `docker run -p 5000:5000 ml-model-service`

## API Endpoints
- `/predict`: Make model predictions
- `/health`: Check service status

## Requirements
- Python 3.9+
- Docker