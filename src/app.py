from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the model
model = joblib.load('best_model.joblib')

# Iris class mapping
IRIS_CLASSES = {
    0: 'Iris-Setosa',
    1: 'Iris-Versicolor', 
    2: 'Iris-Virginica'
}

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.get_json()['data']
        
        # Convert input to numpy array
        input_data = np.array(data).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Get the class name
        class_name = IRIS_CLASSES[prediction[0]]
        
        # Return the prediction
        return jsonify({
            'prediction_code': int(prediction[0]),
            'prediction_class': class_name,
            'input_features': data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)