#%%
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained linear regression model from the .pkl file
# Ensure 'model.pkl' is in the exact same directory as app.py
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    """Renders the main prediction page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles form submission, processes input, and returns prediction."""
    try:
        # Get the experience value from the HTML form input
        experience_input = request.form['experience']
       
        # Convert input to float and reshape into a 2D array (required by scikit-learn models)
        features = np.array([[float(experience)]])
       
        # Make prediction using the loaded model
        prediction = model.predict(features)
       
        # Format output to 2 decimal places
        output = round(prediction[0], 2)
       
        return render_template('index.html', prediction_text=f"Estimated Salary: ${output:,.2f}")
   
    except Exception as e:
        return render_template('index.html', prediction_text="Error: Please enter a valid number for experience.")

if __name__ == "__main__":
    app.run(debug=True)
# %%
