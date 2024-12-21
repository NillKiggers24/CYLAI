from sklearn.preprocessing import StandardScaler
from importlib.metadata import version
from flask import send_from_directory
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
from models import db, User, Prediction
import os  # Добавьте импорт os здесь
# Путь к файлам
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'weight_loss_model.keras'), compile=False)
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Weightloss.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'your_secret_key'
# Инициализация базы данных и Flask-Login
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Загрузка модели
# Перезагрузка старой модели
model = tf.keras.models.load_model('weight_loss_model.keras', compile=False)

# Повторная компиляция с правильными метриками
model.compile(optimizer='adam', loss='mae', metrics=['mean_absolute_error'])

# Сохранение исправленной модели
model.save('corrected_weight_loss_model.keras')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('aboutus.html')

@app.route('/profile_before')
def profile_placeholder():
    return render_template('profile_before.html')


@app.route('/result')
def result():
    prediction = session.get('prediction_result', None)
    if not prediction:
        flash("No prediction available. Please submit your data first.", "error")
        return redirect(url_for('profile_placeholder'))
    # Получаем данные из сессии
    current_weight = prediction['weight']
    target_weight = prediction['target_weight']
    predicted_days = prediction['predicted_days']
    body_fat = prediction['body_fat']

    # Рассчитываем чистую массу
    lean_body_mass = current_weight * (1 - body_fat / 100)  # Чистая масса без жира
    BMR = int(lean_body_mass * 24)  # Базовый обмен веществ (калории в день для поддержания)
    
    # Для похудения нужно снизить калорийность на дефицит, рассчитанный моделью
    calories_needed_for_loss = round(BMR, 2)

    return render_template('result.html', 
                           prediction=prediction, 
                           calories_needed_for_loss=calories_needed_for_loss)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('login'))

        new_user = User(username=username, password=generate_password_hash(password, method='pbkdf2:sha256'))
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful', 'success')
        return redirect(url_for('index'))

    return render_template('login.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Получаем данные из запроса
        data = request.json
        print("Received data:", data)  # Логирование для отладки

        # Преобразование имен признаков
        data_transformed = {
            "Height": data['height'],
            "Weight": data['weight'],
            "Body Fat %": data['bodyFat'],
            "Calorie Deficit": data['calorieDeficit'],
            "Target Weight": data['targetWeight'],
            "Gender": data['gender']  
        }

        # Преобразуем в DataFrame
        new_data = pd.DataFrame([data_transformed])
        print("Transformed DataFrame:", new_data)  # Логирование

        # Нормализация данных
        scaled_data = scaler.transform(new_data)
        print("Scaled data for model:", scaled_data)  # Логирование

        # Получаем предсказание
        prediction = model.predict(scaled_data)
        predicted_days = round(float(prediction[0][0]), 2)
        # Сохраняем результат в сессию
        session['prediction_result'] = {
            'height': data['height'],
            'weight': data['weight'],
            'body_fat': data['bodyFat'],
            'calorie_deficit': data['calorieDeficit'],
            'target_weight': data['targetWeight'],
            'gender': data['gender'],
            'predicted_days': predicted_days
        }
                # Сохранение в базу данных
        pred = Prediction(
            user_id=None,
            height=data['height'],
            weight=data['weight'],
            body_fat=data['bodyFat'],
            calorie_deficit=data['calorieDeficit'],
            target_weight=data['targetWeight'],
            gender=data['gender'],
            predicted_days=predicted_days
        )
        print("Saving to database...")
        db.session.add(pred)
        db.session.commit()
        print("Data saved successfully!")
        return jsonify({'redirect_url': url_for('result')})


    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("Database created at:", app.config['SQLALCHEMY_DATABASE_URI'])
    app.run(debug=True)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)
