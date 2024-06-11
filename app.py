from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS
import pickle

# Load the model from a file
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://127.0.0.1:5500"}})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Preprocess the data
    df = pd.DataFrame(data, index=[0])
    X = pd.get_dummies(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Make predictions
    prediction = model.predict(X_scaled)[0]

    # Convert prediction to a standard Python data type
    prediction = float(prediction)

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True)
