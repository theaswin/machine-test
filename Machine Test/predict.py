from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load model
model = joblib.load("./best.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    data = request.get_json(force=True)
    try:
        features = np.array([
            [
                data['Pregnancies'],
                data['Glucose'],
                data['BloodPressure'],
                data['BMI'],
                data['Age']
            ]
        ])
        prediction = model.predict(features)[0]
        result = "Unhealthy indivitual" if prediction == 1 else "Healthy Indivitual"
        return jsonify({"prediction": int(prediction), "outcome": result})
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    try:
        features = np.array([
            [
                data['Pregnancies'],
                data['Glucose'],
                data['BloodPressure'],
                data['BMI'],
                data['Age']
            ]
        ])
        prediction = model.predict(features)[0]
        result = "Has Diabetes" if prediction == 1 else "No Diabetes"
        return jsonify({"prediction": int(prediction), "outcome": result})
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400


if __name__ == '__main__':
    app.run(debug=True)
