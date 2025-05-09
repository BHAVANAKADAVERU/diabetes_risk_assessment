import os
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
app = Flask(__name__)


app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

model = joblib.load('diabetes_model.pkl')  # Load the ML model

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(300), nullable=False)
    results = db.relationship('RiskRecord', backref='user', lazy=True)

class RiskRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Float)
    gender = db.Column(db.String(10))
    bmi = db.Column(db.Float)
    blood_pressure = db.Column(db.Float)
    glucose = db.Column(db.Float)
    family_history = db.Column(db.Integer)
    past_high_blood_sugar = db.Column(db.Integer)
    hypertension = db.Column(db.Integer)
    gestational_diabetes = db.Column(db.Integer)
    pcos = db.Column(db.Integer)
    physical_activity = db.Column(db.String(20))
    smoking = db.Column(db.Integer)
    alcohol = db.Column(db.Integer)
    prediction_result = db.Column(db.String(20))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.DateTime, default=db.func.now())

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('welcome'))

@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return redirect(url_for('signup'))

        user = User(username=username, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Account created successfully!', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials', 'danger')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('welcome'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=current_user.username)



@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        # Collect input data
        input_data = {
            "Age": float(request.form['Age']),
            "Gender": request.form['Gender'],
            "BMI": float(request.form['BMI']),
            "BloodPressure": float(request.form['BloodPressure']),
            "Glucose": float(request.form['Glucose']),
            "FamilyHistory": int(request.form['FamilyHistory']),
            "PastHighBloodSugar": int(request.form['PastHighBloodSugar']),
            "Hypertension": int(request.form['Hypertension']),
            "GestationalDiabetes": int(request.form['GestationalDiabetes']),
            "PCOS": int(request.form['PCOS']),
            "PhysicalActivity": request.form['PhysicalActivity'],
            "Smoking": int(request.form['Smoking']),
            "Alcohol": int(request.form['Alcohol']),
        }

        # Convert input data to DataFrame for prediction
        input_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = model.predict(input_df)[0]
        result = "High Risk" if prediction == 1 else "Low Risk"

        # Create a new RiskRecord to save the result
        risk_record = RiskRecord(
            age=input_data["Age"],
            gender=input_data["Gender"],
            bmi=input_data["BMI"],
            blood_pressure=input_data["BloodPressure"],
            glucose=input_data["Glucose"],
            family_history=input_data["FamilyHistory"],
            past_high_blood_sugar=input_data["PastHighBloodSugar"],
            hypertension=input_data["Hypertension"],
            gestational_diabetes=input_data["GestationalDiabetes"],
            pcos=input_data["PCOS"],
            physical_activity=input_data["PhysicalActivity"],
            smoking=input_data["Smoking"],
            alcohol=input_data["Alcohol"],
            prediction_result=result,
            user_id=current_user.id  # Link to the logged-in user
        )

        # Add to the database
        db.session.add(risk_record)
        db.session.commit()

        # Pass result to the template
        return render_template('results.html', result=result)
    
    return render_template('index.html')

@app.route('/try_assessment', methods=['GET', 'POST'])
def try_assessment():
    if request.method == 'POST':
        # Collect input data from the form
        input_data = {
            "Age": float(request.form['Age']),
            "Gender": request.form['Gender'],
            "BMI": float(request.form['BMI']),
            "BloodPressure": float(request.form['BloodPressure']),
            "Glucose": float(request.form['Glucose']),
            "FamilyHistory": int(request.form['FamilyHistory']),
            "PastHighBloodSugar": int(request.form['PastHighBloodSugar']),
            "Hypertension": int(request.form['Hypertension']),
            "GestationalDiabetes": int(request.form['GestationalDiabetes']),
            "PCOS": int(request.form['PCOS']),
            "PhysicalActivity": request.form['PhysicalActivity'],
            "Smoking": int(request.form['Smoking']),
            "Alcohol": int(request.form['Alcohol']),
        }

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Define column order based on the model's training data
        model_columns = [
            "Age", "Gender", "BMI", "BloodPressure", "Glucose", 
            "FamilyHistory", "PastHighBloodSugar", "Hypertension", 
            "GestationalDiabetes", "PCOS", "PhysicalActivity", 
            "Smoking", "Alcohol"
        ]

        # Ensure input_df has the same columns in the same order as the model
        input_df = input_df[model_columns]

        # Make prediction
        prediction = model.predict(input_df)[0]
        result = "High Risk" if prediction == 1 else "Low Risk"

        # Return the result to the template
        return render_template('results.html', result=result)

    return render_template('try_assessment.html')



@app.route('/index')
@login_required
def index():
    return render_template('index.html')

@app.route('/results_history')
@login_required
def results_history():
    user_results = RiskRecord.query.filter_by(user_id=current_user.id).all()
    return render_template('results_history.html', results=user_results)

@app.route('/result_detail/<int:id>')
@login_required
def view_result_detail(id):
    record = db.session.query(RiskRecord).filter_by(id=id).first()
    if record:
        return render_template('result_detail.html', record=record)
    else:
        flash('Result not found!', 'danger')
        return redirect(url_for('results_history'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
