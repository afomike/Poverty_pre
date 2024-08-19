from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

app = Flask(__name__)

# Load models and scalers
with open('models/models.pkl', 'rb') as model_file:
    models = pickle.load(model_file)
with open('models/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
with open('models/label_encoders.pkl', 'rb') as encoder_file:
    label_encoders = pickle.load(encoder_file)

def poverty_level(income):
    if income < 15000:
        return 'Below Poverty Line'
    elif 15000 <= income < 50000:
        return 'Middle Class'
    else:
        return 'Above Poverty Line'

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

    # Scale features
    df[['Age', 'Income']] = df[['Age', 'Income']].astype(float)
    df[['Age', 'Income']] = scaler.transform(df[['Age', 'Income']])

    # Predict with one model (e.g., the first model in the dictionary)
    selected_model = list(models.values())[1]  # Choose the desired model
    prediction = selected_model.predict(df)[0]

    # Map prediction to human-readable label
    outcome = poverty_level(prediction)

    return jsonify({'result': outcome})

if __name__ == '__main__':
    app.run(debug=True)
