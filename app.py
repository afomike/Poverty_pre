from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import joblib

app = Flask(__name__)

# Load the pre-trained model and encoders
model = joblib.load('models/random_forest.pkl')  # Load a single model
label_encoders = joblib.load('models/label_encoders.pkl')  # Load encoders as a dictionary


def poverty_level(income):
    if income == 1:
        return 'Below Poverty Line'
    elif income == 2:
        return 'Middle Class'
    elif income == 0:
        return 'Above Poverty Line'
    else:
        return 'unconcluded'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    df = pd.DataFrame([data])

    # Transform the input data using the stored encoders
    label_encode_features = ['Education', 'Employment']
    for column in label_encode_features:
        if column in df.columns:
            le = label_encoders.get(column)
            if le:
                known_classes = set(le.classes_)
                df[column] = df[column].apply(lambda x: le.transform([x])[0] if x in known_classes else -1)

    

    # Predict using the loaded model
    prediction = model.predict(df)[0]

    # Ensure the prediction is an integer for the poverty_level function
    prediction = int(prediction)
    
    # Map prediction to human-readable label
    outcome = poverty_level(prediction)

    return jsonify({'result': outcome})







if __name__ == '__main__':
    app.run(debug=True)
