from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

# Модель пользователя
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Модель истории запросов
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=True)  # Поле для анонимных пользователей
    height = db.Column(db.Float, nullable=False)
    weight = db.Column(db.Float, nullable=False)
    body_fat = db.Column(db.Float, nullable=False)
    calorie_deficit = db.Column(db.Float, nullable=False)
    target_weight = db.Column(db.Float, nullable=False)
    gender = db.Column(db.String(1), nullable=False)
    predicted_days = db.Column(db.Float, nullable=False)