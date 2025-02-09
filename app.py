import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

# Load the trained model
model_path = "C:/Users/mdhwa/Downloads/render-demo/fraud_detection_model.pkl"
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

# Define the expected feature names
FEATURE_NAMES = ['CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS',
                 'TX_FRAUD_SCENARIO', 'TX_HOUR', 'TX_DAY', 'TX_MONTH', 'HIGH_TX_AMOUNT',
                 'CUSTOMER_FRAUD_COUNT', 'TERMINAL_FRAUD_COUNT']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values and ensure they match the expected feature count
        input_features = [float(request.form[feature]) for feature in FEATURE_NAMES]
        
        if len(input_features) != 12:
            return jsonify({"error": f"Expected 12 features, but got {len(input_features)}. Please enter all required fields."})
        
        final_features = [np.array(input_features)]

        # Make prediction
        prediction = model.predict(final_features)
        output = "Fraudulent Transaction" if prediction[0] == 1 else "Legitimate Transaction"

        return render_template('index.html', prediction_text=f'Prediction: {output}')
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
