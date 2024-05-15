import pickle
from flask import Flask, render_template, request
import numpy as np
import pandas as pd

# Flask web application
app = Flask(__name__)

# Load model from the pickle file
model_file = 'species-svm.pickle'
with open(model_file, 'rb') as file:
    model_data = pickle.load(file)
    
model = model_data['model']
scaler = model_data['scaler']
species_map = model_data['species_map']

# Route for the root URL ("/")
@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_species = None
    if request.method == 'POST':
        SepalLengthCm = request.form.get('SepalLengthCm')
        SepalWidthCm = request.form.get('SepalWidthCm')
        PetalLengthCm = request.form.get('PetalLengthCm')
        PetalWidthCm = request.form.get('PetalWidthCm')
        
        # input data to list
        input_data = [[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]]
        
        # use dataframe to add feature names/column
        X = pd.DataFrame(
            input_data,
            columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        )
        
        # transform using fitted StandardScaler
        x_scaled = scaler.transform(X)
        
        # use model to predict
        y_preds = model.predict(x_scaled)
        y_pred = y_preds[0]
        
        # map predicted species id to species name
        predicted_species = species_map[y_pred]
        print('Hasil Prediksi', predicted_species)

    return render_template(
        'index.html',
        PRED_RESULT=predicted_species
    )

if __name__ =='__main__':
    # Run the application on a local development server
    app.run(debug=True)