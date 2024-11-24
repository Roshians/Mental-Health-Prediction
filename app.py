from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the saved model, scaler, and feature names
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('scaler.joblib')
feature_names = joblib.load('feature_names.joblib')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        form_data = request.form.to_dict()
        
        # Handle Sleep Duration
        sleep_duration = float(form_data.get('Sleep Duration', 0))
        form_data['Sleep Duration_6-8 hours'] = 1 if 6 <= sleep_duration <= 8 else 0

        # Create a numpy array with the input features
        input_features = np.array([float(form_data.get(feature, 0)) for feature in feature_names]).reshape(1, -1)

        # Scale the input features
        scaled_features = scaler.transform(input_features)

        # Make prediction
        prediction = model.predict(scaled_features)

        # Get prediction probability
        prediction_proba = model.predict_proba(scaled_features)[0][1]

        # Prepare the response
        result = {
            'prediction': 'Depression' if prediction[0] == 1 else 'No Depression',
            'probability': f'{prediction_proba:.2f}'
        }

        return render_template('index.html', result=result)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

