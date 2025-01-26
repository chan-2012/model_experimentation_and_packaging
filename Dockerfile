# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the entire project directory
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure the model and app are copied
COPY src/app.py /app/app.py
COPY models/best_model.joblib /app/best_model.joblib

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]