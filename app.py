from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model
model_path = os.path.join("models", "student_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs
        study_hours = float(request.form['study_hours'])
        attendance = float(request.form['attendance'])
        gender = int(request.form['gender'])          # 0 for Male, 1 for Female
        parental_edu = int(request.form['parental_edu'])  # 0,1,2

        # Predict
        features = np.array([[study_hours, parental_edu, attendance, gender]])
        prediction = model.predict(features)[0]

        return render_template('index.html', prediction_text=f"Predicted Score: {prediction:.2f}")
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
