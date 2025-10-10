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
    'alcohol_consumption': LabelEncoder().fit(['None', 'High', 'Moderate', 'Low']),
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
    with open('random_forest_model.pkl', 'rb') as f:
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
        
        # Encode categorical features
        features = [
            float(data['age']),
            label_encoders['gender'].transform([data['gender']])[0],
            int(data['hypertension']),
            int(data['diabetes']),
            float(data['cholesterol']),
            int(data['obesity']),
            float(data['waist_circumference']),
            label_encoders['family_history'].transform([data['family_history']])[0],
            label_encoders['smoking_status'].transform([data['smoking_status']])[0],
            label_encoders['alcohol_consumption'].transform([data['alcohol_consumption']])[0],
            label_encoders['physical_activity'].transform([data['physical_activity']])[0],
            label_encoders['dietary_habits'].transform([data['dietary_habits']])[0],
            label_encoders['stress_level'].transform([data['stress_level']])[0],
            label_encoders['sleep_hours_per_day'].transform([data['sleep_hours_per_day']])[0],
            float(data['blood_pressure']),
            float(data['blood_glucose']),
            float(data['fasting_blood_sugar']),
            float(data['cholesterol_total']),
            int(data['blood_pressure_medication']),
            int(data['cholesterol_medication']),
            int(data['triglyceride_medication']),
            label_encoders['EKG_results'].transform([data['EKG_results']])[0],
            int(data['previous_heart_problems'])
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