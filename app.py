from flask import Flask, request, jsonify, render_template, redirect, url_for
import joblib
import pandas as pd
import numpy as np
import pickle, os, re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from flask_cors import CORS
from sklearn.linear_model import LogisticRegression

app = Flask(__name__, static_url_path='/')
CORS(app)
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": ["Content-Type", "Authorization"]}})

# Define maxlen
maxlen = 5
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_fraudulent(data,scaler):
    # Fill missing age
    #Predefined mean age (computed from training set)
    mean_age = 35.0  # Example value, replace with actual mean from training datas
    data['Customer Age'] = data['Customer Age'].apply(lambda x: mean_age if x < 10 else x)
    
    # Address Match
    data['Address Match'] = (data['Shipping Address'] == data['Billing Address']).astype(int)
    
    # Drop unnecessary columns
    data = data.drop(["Transaction ID", "Customer ID", "Customer Location", "Transaction Date", 
                      "IP Address", "Shipping Address", "Billing Address"], axis=1)
    
    # Map categorical features
    data['Payment Method'] = data['Payment Method'].map({"debit card": 0, "credit card": 1, "PayPal": 2, "bank transfer": 3})
    data['Product Category'] = data['Product Category'].map({"home & garden": 0, "electronics": 1, "toys & games": 2, 
                                                             "clothing": 3, "health & beauty": 4})
    data['Device Used'] = data['Device Used'].map({"desktop": 0, "mobile": 1, "tablet": 2})
    
    # Scale numeric features
    numeric_features = ['Transaction Amount', 'Quantity', 'Customer Age', 'Account Age Days', 'Transaction Hour']
    data[numeric_features] = scaler.transform(data[numeric_features])
    
    return data


def preprocess_sentiment(text):
    loaded_model=joblib.load("models/sentiment.pkl")
    loaded_stop=joblib.load("models/stopwords.pkl")
    loaded_vec=joblib.load("models/vectorizer.pkl")
    label = {0: 'negative', 1: 'positive'}
    X = loaded_vec.transform([text])
    y = loaded_model.predict(X)[0]
    proba = np.max(loaded_model.predict_proba(X))
    return label[y], proba


@app.route('/predict-fraudulent', methods=['POST'])
def predict_fraudulent():
    try:
        # Load the saved data from the pickle file
        with open('models/fraud_detection_data.pkl', 'rb') as file:
            saved_data = pickle.load(file)
            model = saved_data['model']
            scaler = saved_data['scaler']
            fraud_map = saved_data['fraud_map']
        # Parse form data with reordered keys
        data = {
            "Transaction Amount": float(request.form.get("Transaction Amount")),
            "Payment Method": request.form.get("Payment Method"),
            "Product Category": request.form.get("Product Category"),
            "Quantity": int(request.form.get("Quantity")),
            "Customer Age": int(request.form.get("Customer Age")),
            "Device Used": request.form.get("Device Used"),
            "Address Match": (request.form.get("Address Match")),
            "Account Age Days": (request.form.get("Account Age Days")),
            "Transaction Hour": int(request.form.get("Transaction Hour")),
            "Transaction ID": request.form.get("Transaction ID"),
            "Customer ID": request.form.get("Customer ID"),
            "Customer Location": request.form.get("Customer Location"),
            "Transaction Date": request.form.get("Transaction Date"),
            "IP Address": request.form.get("IP Address"),
            "Shipping Address": request.form.get("Shipping Address"),
            "Billing Address": request.form.get("Billing Address")
        }

        # Convert data to DataFrame
        df = pd.DataFrame([data])

        # Preprocess the data
        preprocessed_data = preprocess_fraudulent(df,scaler)

        # Make predictions
        predictions = model.predict(preprocessed_data)

        # Create response
        result = {"is_fraudulent": int(predictions[0])}
        return render_template('result_fraudulent.html', result=result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict-sentiment', methods=['POST'])
def predict_sentiment():
    # try:
        
        # Parse form data
        text = request.form.get("Review Text")

        # Check if preprocessed_text is empty
        if not text:
            error_message = "Please provide a valid review text."
            return render_template('result-sentiment.html', error_message=error_message)

        prediction, proba = preprocess_sentiment(text)
        # Create an alert message based on prediction
        if prediction == 'positive':
            alert_type = 'success'
            alert_message = 'The review is Positive.'
        else:
            alert_type = 'danger'
            alert_message = 'The review is Negative.'

        print(alert_message)

        # Render the result in the HTML with a Bootstrap alert
        return render_template('result_sentiment.html', alert_type=alert_type, alert_message=alert_message)
    # except Exception as e:
    #     return str(e)


@app.route('/predict-fraudulent-form', methods=['GET'])
def predict_fraudulent_form():
    return render_template('predict_fraudulent_form.html')

@app.route('/predict-sentiment-form', methods=['GET'])
def predict_sentiment_form():
    return render_template('predict_sentiment_form.html')



if __name__ == '__main__':
    app.run(debug=True, port=5000)
