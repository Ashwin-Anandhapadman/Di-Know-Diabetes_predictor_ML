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
ridge_model = pickle.load(open(r'models2/diabetes2_ridge.pkl', 'rb'))
scaler_model = pickle.load(open(r'models2/diabetes2_scaler.pkl', 'rb'))
elastc_model = pickle.load(open(r'models2/diabetes_ene.pkl', 'rb'))
ann_model = load_model(r'D:/Ash_ML/ML_web/Diabetes_proj/models2/model.h5', custom_objects={'mse': MeanSquaredError})

@app.route("/") 
def index():
    return render_template('index_diab.html')

@app.route('/predict', methods=['GET', 'POST']) 
def predict():
    if request.method == "POST":
        try:
            # Get input values from the form
            Insulin = request.form.get('Insulin')
            DiabetesPedigreeFunction = request.form.get('DiabetesPedigreeFunction')
            Pregnancies = request.form.get('Pregnancies')
            Age = request.form.get('Age')
            BMI = request.form.get('BMI')
            Glucose = request.form.get('Glucose')

            # Check for None and convert to float or set default value
            Insulin = float(Insulin) if Insulin else 0.0
            DiabetesPedigreeFunction = float(DiabetesPedigreeFunction) if DiabetesPedigreeFunction else 0.0
            Pregnancies = float(Pregnancies) if Pregnancies else 0.0
            Age = float(Age) if Age else 0.0
            BMI = float(BMI) if BMI else 0.0
            Glucose = float(Glucose) if Glucose else 0.0

            # Prepare data for prediction (reshape into 2D array)
            new_data = np.array([[Insulin, DiabetesPedigreeFunction, Pregnancies, Age, BMI, Glucose]])
            new_data_scaled = scaler_model.transform(new_data)

            # Predict using all models
            ann_prediction = ann_model.predict(new_data_scaled)
            ridge_prediction = ridge_model.predict(new_data_scaled)
            elastc_prediction = elastc_model.predict(new_data_scaled)

            # Convert predictions to binary outcomes (e.g., 0 or 1) for voting
            ann_result = (ann_prediction > 0.5).astype(int)
            ridge_result = (ridge_prediction > 0.5).astype(int)
            elastc_result = (elastc_prediction > 0.5).astype(int)

            # Aggregate results using majority voting
            final_prediction = np.array([ann_result[0][0], ridge_result[0], elastc_result[0]])
            vote_count = np.bincount(final_prediction.astype(int))
            max_vote_class = np.argmax(vote_count)

            return render_template('home_diab.html', results=max_vote_class)
        except Exception as e:
            return str(e)  # Simple error message for debugging
    else:
        return render_template("home_diab.html")

if __name__ == "__main__": 
    app.run(host="0.0.0.0")
