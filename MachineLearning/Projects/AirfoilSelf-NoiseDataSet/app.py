import pickle
from flask import Flask, render_template, request, app, jsonify
import numpy as np
import pandas as pd

import sklearn

print('The scikit-learn version is {}.'.format(sklearn.__version__))

## Prediction for Airfoil Self-Noise Data Set : https://archive.ics.uci.edu/ml/datasets/airfoil+self-noise

app=Flask(__name__)
output = 0
algorithm = ''

def pick_a_model(algorithm):
    if algorithm == "DecisionTree":
        return pickle.load(open('./models/decisiontree_regressor_model.pkl', 'rb'))
    if algorithm == "ElasticnetRegressor":
        return pickle.load(open('./models/elasticnet_regressor_model.pkl', 'rb'))
    if algorithm == "LinearRegressor":
        return pickle.load(open('./models/linear_regressor_model.pkl', 'rb'))
    if algorithm == "LassoRegressor":
        return pickle.load(open('./models/lasso_regressor_model.pkl', 'rb'))
    if algorithm == "RidgeRegressor":
        return pickle.load(open('./models/ridge_regressor_model.pkl', 'rb'))
    if algorithm == "SupportVectorRegressor":
        return pickle.load(open('./models/svr_regressor_model.pkl', 'rb'))
    if algorithm == "RandomForest":
        return pickle.load(open('./models/randomforest_regressor_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html', output=output, algorithm=algorithm)

@app.route('/predict', methods=['POST'])
def predict():
    data=[x for x in request.form.values()]
    print(data)
    algorithm = data[0]
    model = pick_a_model(algorithm)
    final_features = [[float(i) for i in data[1:]]]
    print(final_features)
    output = model.predict(final_features)[0]
    print(output)
    return render_template('home.html', output=output, algorithm=algorithm)

if __name__=="__main__":
    app.run(debug=True)
