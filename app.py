from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError

application = Flask(__name__)
app = application

# Load models with proper handling for custom metrics if needed
svm_model = pickle.load(open(r'models3/diabetes3_svm.pkl', 'rb'))
scaler_model = pickle.load(open(r'models3/diabetes3_scaler.pkl', 'rb'))
elastc_model = pickle.load(open(r'models3/diabetes3_ene.pkl', 'rb'))
ann_model = load_model(r'D:/Ash_ML/ML_web/Diabetes_proj/models3/diabetes3_ann.h5')

@app.route("/") 
def index():
    return render_template('index_diab.html')

@app.route('/predict', methods=['GET', 'POST']) 
def predict():
    if request.method == "POST":
        try:
            # Get input values from the form
            Insulin = float(request.form.get('Insulin', 0.0))  # Default to 0.0 if None
            DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction', 0.0))
            Pregnancies = float(request.form.get('Pregnancies', 0.0))
            Age = float(request.form.get('Age', 0.0))
            BMI = float(request.form.get('BMI', 0.0))
            BloodPressure = float(request.form.get('BloodPressure', 0.0))
            Glucose = float(request.form.get('Glucose', 0.0))

            # Prepare data for prediction (reshape into 2D array)
            new_data = np.array([[Insulin, DiabetesPedigreeFunction, Pregnancies, Age, BMI,BloodPressure, Glucose]])
            new_data_scaled = scaler_model.transform(new_data)

            # Predict using all models
            ann_prediction = ann_model.predict(new_data_scaled)
            svm_prediction = svm_model.predict(new_data_scaled)
            elastc_prediction = elastc_model.predict(new_data_scaled)

            # Convert predictions to binary outcomes (e.g., 0 or 1) for voting
            ann_result = (ann_prediction > 0.5).astype(int)  # Assuming binary classification
            svm_result = (svm_prediction > 0.5).astype(int)
            elastc_result = (elastc_prediction > 0.5).astype(int)

            # Aggregate results using majority voting
            final_prediction = np.array([ann_result[0][0], svm_result[0], elastc_result[0]])
            vote_count = np.bincount(final_prediction.astype(int))
            max_vote_class = np.argmax(vote_count)  # Get class with maximum votes

            # Map numeric result to string
            result_label = "Diabetic !" if max_vote_class == 1 else "Non-Diabetic !"

            return render_template('home_diab.html', results=result_label)
        except Exception as e:
            return str(e)  # Simple error message for debugging
    else:
        return render_template("home_diab.html")


if __name__ == "__main__": 
    app.run(host="0.0.0.0", debug=True)
