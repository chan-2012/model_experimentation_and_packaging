from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the model
model = joblib.load('best_model.joblib')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.json['data']
        
        # Convert input to numpy array
        input_data = np.array(data).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Return the prediction
        return jsonify({
            'prediction': int(prediction[0])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)