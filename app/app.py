from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Initialize label encoders with your dataset's unique values
# These should match the order and values from your training data
label_encoders = {
    'gender': LabelEncoder().fit(['Male', 'Female']),
    'hypertension': LabelEncoder().fit([0, 1]),
    'diabetes': LabelEncoder().fit([0, 1]),
    'obesity': LabelEncoder().fit([0, 1]),
    'family_history': LabelEncoder().fit(['Never', 'Past', 'Current']),
    'smoking_status': LabelEncoder().fit(['None', 'High', 'Moderate', 'Low']),
    'alcohol_consumption': LabelEncoder().fit(['None', 'Low', 'Moderate', 'High', 'Unknown']),
    'physical_activity': LabelEncoder().fit(['Low', 'Moderate', 'High']),
    'dietary_habits': LabelEncoder().fit(['Unhealthy', 'Healthy']),
    'stress_level': LabelEncoder().fit(['Low', 'Moderate', 'High']),
    'sleep_hours_per_day': LabelEncoder().fit(['Poor', 'Normal', 'Good']),
    'blood_pressure_medication': LabelEncoder().fit([0, 1]),
    'cholesterol_medication': LabelEncoder().fit([0, 1]),
    'triglyceride_medication': LabelEncoder().fit([0, 1]),
    'EKG_results': LabelEncoder().fit(['Normal', 'Abnormal']),
    'previous_heart_problems': LabelEncoder().fit([0, 1])
}

# Load your trained model
try:
    with open('../models/FC212013_Maleesha/xgboost_best.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✓ Model loaded successfully!")
except Exception as e:
    model = None
    print(f"Warning: Model not found. Error: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        features = [
            float(data['age']),                                           # age
            int(data['hypertension']),                                    # hypertension
            int(data['diabetes']),                                        # diabetes
            float(data['cholesterol']),                                   # cholesterol_level
            int(data['obesity']),                                         # obesity
            float(data['waist_circumference']),                           # waist_circumference
            label_encoders['family_history'].transform([data['family_history']])[0],  # family_history
            label_encoders['sleep_hours_per_day'].transform([data['sleep_hours_per_day']])[0],  # sleep_hours
            float(data['blood_pressure']),                                # Systolic BP
            float(data['diastolic_bp']) if 'diastolic_bp' in data else 80.0,  # Diastolic BP (optional)
            float(data['fasting_blood_sugar']),                           # fasting_blood_sugar
            float(data['cholesterol_hdl']) if 'cholesterol_hdl' in data else 50.0,  # HDL (optional)
            float(data['cholesterol_ldl']) if 'cholesterol_ldl' in data else 120.0, # LDL (optional)
            float(data['triglycerides']) if 'triglycerides' in data else 150.0,     # triglycerides (optional)
            int(data['previous_heart_problems']),                          # previous_heart_disease
            int(data['blood_pressure_medication']) or
            int(data['cholesterol_medication']) or
            int(data['triglyceride_medication']),                         # medication_usage (combined)
            1 if data['gender'] == 'Male' else 0,                         # gender_Male
            1 if data['smoking_status'] == 'None' else 0,                 # smoking_status_Never
            1 if data['smoking_status'] == 'Low' else 0,                  # smoking_status_Past
            1 if data['alcohol_consumption'] == 'Moderate' else 0,        # alcohol_consumption_Moderate
            1 if data['alcohol_consumption'] == 'None' else 0,            # alcohol_consumption_unknown
            1 if data['physical_activity'] == 'Low' else 0,               # physical_activity_Low
            1 if data['physical_activity'] == 'Moderate' else 0,          # physical_activity_Moderate
            1 if data['dietary_habits'] == 'Unhealthy' else 0,            # dietary_habits_Unhealthy
            1 if data['stress_level'] == 'Low' else 0,                    # stress_level_Low
            1 if data['stress_level'] == 'Moderate' else 0,               # stress_level_Moderate
            1 if data['EKG_results'] == 'Normal' else 0                   # EKG_results_Normal
        ]

        # Convert to numpy array
        features_array = np.array(features).reshape(1, -1)

        # Make prediction
        if model:
            prediction = model.predict(features_array)[0]
            probability = model.predict_proba(features_array)[0]

            result = {
                'prediction': int(prediction),
                'probability': float(probability[1] * 100),
                'risk_level': 'High' if probability[1] > 0.6 else 'Moderate' if probability[1] > 0.3 else 'Low'
            }
        else:
            # Demo response if model not loaded
            result = {
                'prediction': 0,
                'probability': 35.5,
                'risk_level': 'Moderate',
                'demo': True
            }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    print("Starting Flask server...")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True)
