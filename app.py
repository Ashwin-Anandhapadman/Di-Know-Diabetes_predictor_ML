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
#svm_model = pickle.load(open(r'models3/diabetes3_svm.pkl', 'rb'))
scaler_model = pickle.load(open(r'models3/diabetes3_scaler.pkl', 'rb'))
lr_model = pickle.load(open(r'models3/diabetes3_lr.pkl', 'rb'))
xgb_model = pickle.load(open(r'models3/diabetes3_xgb.pkl', 'rb'))
ann_model = load_model(r'D:/Ash_ML/ML_web/Diabetes_proj/models3/diabetes3_ann.h5')

@app.route("/") 
def index():
    return render_template('index_diab.html')

@app.route('/predict', methods=['GET', 'POST']) 
def predict():
    if request.method == "POST":
        try:
            
            Insulin = float(request.form.get('Insulin', 0.0))  # Default to 0.0 if None
            DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction', 0.0))
            Pregnancies = float(request.form.get('Pregnancies', 0.0))
            Age = float(request.form.get('Age', 0.0))
            BMI = float(request.form.get('BMI', 0.0))
            BloodPressure = float(request.form.get('BloodPressure', 0.0))
            Glucose = float(request.form.get('Glucose', 0.0))
            SkinThickness = float(request.form.get('SkinThickness', 0.0))



            new_data = np.array([[Insulin, DiabetesPedigreeFunction, Pregnancies, Age, BMI, BloodPressure, Glucose, SkinThickness]])
            new_data_scaled = scaler_model.transform(new_data)

      
            ann_prediction = ann_model.predict(new_data_scaled)
            #svm_prediction = svm_model.predict(new_data_scaled)
            lr_prediction = lr_model.predict_proba(new_data_scaled)[:, 1] 
            xgb_prediction = xgb_model.predict_proba(new_data_scaled)[:, 1]  

       
            ann_result = (ann_prediction > 0.55).astype(int)   
            lr_result = (lr_prediction > 0.43).astype(int)  
            xgb_result = (xgb_prediction > 0.50).astype(int)  
            #svm_result = svm_prediction.astype(int)

            final_prediction = np.array([ann_result[0][0], lr_result[0], xgb_result[0]])
            vote_count = np.bincount(final_prediction.astype(int))
            max_vote_class = np.argmax(vote_count)  # Get class with maximum votes

          
            result_label = "Diabetic !" if max_vote_class == 1 else "Non-Diabetic !"

            return render_template('home_diab.html', results=result_label)
        except Exception as e:

            print(f"Error: {e}")
            return render_template('home_diab.html', results="An error occurred during prediction. Please try again.")
    else:
        return render_template("home_diab.html")

if __name__ == "__main__": 
    app.run(host="0.0.0.0", debug=True)
