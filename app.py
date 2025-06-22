from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os


app = Flask(__name__)
CORS(app)

model = joblib.load("house_price_model.pkl")

@app.route('/')
def home():
    return "🏠 House Price Prediction API is live! Use the /predict endpoint."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    binary_map = {'yes': 1, 'no': 0}

    # Convert input to match model format (13 features)
    input_data = [
        float(data['area']),
        int(data['bedrooms']),
        int(data['bathrooms']),
        int(data['stories']),
        binary_map[data['mainroad']],
        binary_map[data['guestroom']],
        binary_map[data['basement']],
        binary_map[data['hotwaterheating']],
        binary_map[data['airconditioning']],
        int(data['parking']),
        binary_map[data['prefarea']],
        1 if data['furnishingstatus'] == 'semi-furnished' else 0,
        1 if data['furnishingstatus'] == 'unfurnished' else 0
    ]

    input_array = np.array([input_data])
    prediction = model.predict(input_array)[0]

    return jsonify({"predicted_price": prediction})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000)) 
    app.run(host='0.0.0.0', port=port)

