from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load('random_forest_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get data from form
        data = request.form.to_dict()

        # Convert input data to numpy array and reshape for model
        input_features = np.array([[
            float(data['gender']),
            float(data['SeniorCitizen']),
            float(data['tenure']),
            float(data['MultipleLines']),
            float(data['InternetService']),
            float(data['Contract']),
            float(data['PaperlessBilling']),
            float(data['MonthlyCharges'])
        ]])
        
        # Predict using the loaded model
        prediction = model.predict(input_features)[0]
        prediction = 'Yes' if prediction == 1 else 'No'
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
