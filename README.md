# Welcome to Di-Know: A smart diabetes predictor 

## About the project:
This project is my attempt at developing a machine learning-based web application called Di-Know. It aims to check for the presence of diabetes in patients by taking few key variables as input. The following variables are used as input for he model:
    1. Pregnancies: Number of times pregnant
    2. Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
    3. Insulin: 2-Hour serum insulin (mu U/ml)
    4. BMI: Body mass index (weight in kg/(height in m)^2)
    5. Age: Age (years)
    5. DiabetesPedigreeFunction: Diabetes pedigree function

If any of the above variables are unknown, kindly provide the input as '0' for that variable.

## The machine learning models:
The ML models used in this study are neural networks, ridge regressor and elastic net regressor. The prediciton results published by these models are taken together into consideration before providing the final result to the user. This is done to ensure that there is no bias in the prediction and more than one model should predict the result before user gets to see it.

The ML models for this project were trained using the Kaggle diabetes dataset: (https://www.kaggle.com/datasets/mathchi/diabetes-data-set).
This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes.





