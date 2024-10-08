from flask import Flask, request, render_template, redirect, url_for, flash, session
import pandas as pd
import joblib
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Simulated user database
user_db = {}

# Load the trained model and label encoders
model = joblib.load('xgboost_model7.joblib')
label_encoders = joblib.load('label_encoders2.joblib')

# Function to convert "Yes" or "No" to 1 or 0
def yes_no_to_numeric(value):
    return 1 if value.lower() == "yes" else 0

@app.route('/')
def onboarding():
    return render_template('onboarding.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        if email in user_db:
            flash('Email already registered. Please sign in.', 'warning')
            return redirect(url_for('signin'))

        user_db[email] = {
            'username': username,
            'password': generate_password_hash(password)
        }
        session['user'] = username
        flash('Sign up successful!', 'success')
        return redirect(url_for('home'))
    return render_template('sign_up.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = user_db.get(email)
        if user and check_password_hash(user['password'], password):
            session['user'] = user['username']
            flash('Sign in successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password. Please try again.', 'danger')
    return render_template('sign_in.html')

@app.route('/home')
def home():
    if 'user' not in session:
        flash('Please sign in to access the prediction form.', 'warning')
        return redirect(url_for('signin'))
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = {
        'Age': int(request.form['age']),
        'Gender': request.form['gender'],
        'Fever': yes_no_to_numeric(request.form['fever']),
        'Cough': yes_no_to_numeric(request.form['cough']),
        'SoreThroat': yes_no_to_numeric(request.form['sorethroat']),
        'ShortnessOfBreath': yes_no_to_numeric(request.form['shortnessofbreath']),
        'Headache': yes_no_to_numeric(request.form['headache']),
        'MusclePain': yes_no_to_numeric(request.form['musclepain']),
        'Fatigue': yes_no_to_numeric(request.form['fatigue']),
        'Nausea': yes_no_to_numeric(request.form['nausea']),
        'Vomiting': yes_no_to_numeric(request.form['vomiting']),
        'Diarrhea': yes_no_to_numeric(request.form['diarrhea']),
        'LossOfTaste': yes_no_to_numeric(request.form['lossoftaste']),
        'LossOfSmell': yes_no_to_numeric(request.form['lossofsmell']),
        'Congestion': yes_no_to_numeric(request.form['congestion']),
        'ChestPain': yes_no_to_numeric(request.form['chestpain']),
        'Chills': yes_no_to_numeric(request.form['chills']),
        'Sweating': yes_no_to_numeric(request.form['sweating']),
        'Rash': yes_no_to_numeric(request.form['rash']),
        'Conjunctivitis': yes_no_to_numeric(request.form['conjunctivitis']),
        'Hospitalized': yes_no_to_numeric(request.form['hospitalized'])
    }

    # Check if all symptoms are "No"
    if all(value == 0 for key, value in user_input.items() if key not in ['Age', 'Gender']):
        return render_template('predict.html', prediction="No disease detected. Please select your symptoms accurately for better prediction.")

    # Convert the input to a DataFrame
    input_data = pd.DataFrame([user_input])

    # Convert gender to numeric
    input_data['Gender'] = input_data['Gender'].map({'Male': 1, 'Female': 0})

    # Predict using the trained model
    prediction = model.predict(input_data)
    disease_map = {
        0: "Influenza", 1: "COVID-19", 2: "Common Cold", 3: "Pneumonia", 4: "Malaria",
        5: "Lassa fever", 6: "Yellow fever", 7: "Cholera", 8: "Tuberculosis",
        9: "HIV", 10: "STD", 11: "HIV/AIDS", 12: "Typhoid fever", 13: "Hepatitis B"
    }

    predicted_disease = disease_map[prediction[0]]

    return render_template('predict.html', prediction=predicted_disease)

if __name__ == '__main__':
    app.run(debug=True)
