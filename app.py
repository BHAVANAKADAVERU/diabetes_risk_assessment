import os
import re
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd
from dotenv import load_dotenv
import shap

# Load environment variables from .env file
load_dotenv()
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'super-secret-dev-key')

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Load model
model = joblib.load('diabetes_model.pkl')  # pipeline: preprocessor + classifier
pipeline = model

# Extract classifier for prediction
classifier = pipeline.named_steps['classifier']
preprocessor = pipeline.named_steps['preprocessor']

# ✅ FIX: Extract RandomForestClassifier from calibrated model
try:
    rf_model_for_shap = classifier.calibrated_classifiers_[0].estimator
    explainer = shap.TreeExplainer(rf_model_for_shap)
except Exception as e:
    raise ValueError("Could not extract base estimator from CalibratedClassifierCV for SHAP.") from e

# -------------------------- Models --------------------------

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

# -------------------------- Routes --------------------------

@app.route('/')
def home():
    return redirect(url_for('dashboard') if current_user.is_authenticated else 'welcome')

@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        if not re.match(r'^[A-Za-z][A-Za-z0-9_]*$', username):
            flash('Username must start with a letter and contain only letters, numbers, or underscores.', 'danger')
            return redirect(url_for('signup'))

        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'danger')
            return redirect(url_for('signup'))

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('signup'))

        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password)
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

        input_df = pd.DataFrame([input_data])
        X_processed = preprocessor.transform(input_df)

        prediction = classifier.predict(X_processed)[0]
        proba = classifier.predict_proba(X_processed)[0][1]
        risk_percentage = round(proba * 100, 2)
        result = "High Risk" if prediction == 1 else "Low Risk"

        shap_values = explainer.shap_values(X_processed)
        numerical = ['Age', 'BMI', 'BloodPressure', 'Glucose']
        categorical = ['Gender', 'PhysicalActivity']
        binary = ['FamilyHistory', 'PastHighBloodSugar', 'Hypertension', 'GestationalDiabetes', 'PCOS', 'Smoking', 'Alcohol']
        cat_encoder = preprocessor.named_transformers_['cat']
        cat_names = cat_encoder.get_feature_names_out(categorical)
        feature_names = list(numerical) + list(cat_names) + list(binary)

        shap_vals_for_instance = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        shap_vals_for_instance = np.array(shap_vals_for_instance).flatten()
        explanation = dict(zip(feature_names, shap_vals_for_instance))
        explanation_sorted = dict(sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True))

        reasons = []
        suggestions = []

        if input_data["BMI"] > 25:
            reasons.append("High BMI")
            suggestions.append("Maintain a healthy weight with balanced diet and exercise.")
        if input_data["Glucose"] > 140:
            reasons.append("High Glucose")
            suggestions.append("Monitor sugar intake and consider a doctor.")
        if input_data["FamilyHistory"]:
            reasons.append("Family history of diabetes")
            suggestions.append("Adopt a preventive lifestyle and regular checkups.")
        if input_data["Hypertension"]:
            reasons.append("High Blood Pressure")
            suggestions.append("Reduce salt and manage stress.")
        if input_data["Smoking"]:
            reasons.append("Smoking habit")
            suggestions.append("Quitting smoking improves insulin sensitivity.")
        if input_data["Alcohol"]:
            reasons.append("Alcohol consumption")
            suggestions.append("Limit alcohol for better metabolic health.")
        if input_data["PhysicalActivity"] == "Low":
            reasons.append("Low physical activity")
            suggestions.append("Incorporate 30 minutes of exercise daily.")

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
            user_id=current_user.id
        )
        db.session.add(risk_record)
        db.session.commit()

        top_shap_features = list(explanation_sorted.items())[:5]
        shap_feature_reasons = []

        for feat, val in top_shap_features:
            if "PhysicalActivity_Low" in feat and val > 0:
                shap_feature_reasons.append("Low physical activity increased your risk.")
            elif "Glucose" in feat and val > 0:
                shap_feature_reasons.append("High glucose level raised your risk.")
            elif "BMI" in feat and val > 0:
                shap_feature_reasons.append("High BMI contributed to your risk.")
            elif "BloodPressure" in feat and val > 0:
                shap_feature_reasons.append("High blood pressure added to your risk.")
            elif "Smoking" in feat and val > 0:
                shap_feature_reasons.append("Smoking increased your risk.")

        return render_template('results.html',
                               result=result,
                               risk_percentage=risk_percentage,
                               reasons=reasons,
                               suggestions=suggestions,
                               shap_explanation=top_shap_features)

    return render_template('index.html')

@app.route('/results_history')
@login_required
def results_history():
    results = RiskRecord.query.filter_by(user_id=current_user.id).all()
    return render_template('results_history.html', results=results)

@app.route('/result_detail/<int:id>')
@login_required
def view_result_detail(id):
    record = RiskRecord.query.filter_by(id=id, user_id=current_user.id).first()
    if record:
        return render_template('result_detail.html', record=record)
    flash('Result not found or access denied!', 'danger')
    return redirect(url_for('results_history'))

@app.route('/index')
@login_required
def index():
    return render_template('index.html')

@app.route('/try_assessment', methods=['GET', 'POST'])
def try_assessment():
    if request.method == 'POST':
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
        input_df = pd.DataFrame([input_data])
        input_df = input_df[[ "Age", "Gender", "BMI", "BloodPressure", "Glucose",
                              "FamilyHistory", "PastHighBloodSugar", "Hypertension",
                              "GestationalDiabetes", "PCOS", "PhysicalActivity",
                              "Smoking", "Alcohol"]]
        prediction = classifier.predict(preprocessor.transform(input_df))[0]
        result = "High Risk" if prediction == 1 else "Low Risk"
        return render_template('try_assess_results.html', result=result)
    return render_template('try_assessment.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
