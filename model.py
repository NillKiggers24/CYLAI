import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*NotOpenSSLWarning.*")
# Функция для расчета дней похудения
def calculate_days_to_lose_weight(weight, height, body_fat_percent, calorie_deficit, target_weight, age, gender):

    lean_body_mass = weight * (1 - body_fat_percent / 100)
    bmr = lean_body_mass * 24 # базовый обмен веществ
    if gender == '1' : bmr *=0.9
    daily_deficit = calorie_deficit + (weight*24 - bmr)
    fat_to_lose = weight - target_weight
    if 13 <= body_fat_percent <= 18:
        fat_calories = fat_to_lose * 5600
    else:
        fat_calories = fat_to_lose * 7700  # калорийность килограмма жира
    return max(1, fat_calories / daily_deficit) if daily_deficit > 0 else None

data_size = 15000
genders = np.random.choice(['0', '1'], data_size)

# Диапазоны для веса и роста по полу
heights = [np.random.randint(160, 191) if g == '0' else np.random.randint(150, 190) for g in genders]
weights = [np.random.randint(60, 140) if g == '0' else np.random.randint(57, 120) for g in genders]

# Генерация процентов жира, типов похудения и целей веса
body_fat_percents = []
for i in range(data_size):
    if weights[i] <= 70:
        body_fat_percents.append(np.random.uniform(13, 25))
    elif genders[i] == '1' and weights[i] > 90: # чтобы не было женских ронни колеманов
        body_fat_percents.append(np.random.uniform(25, 35))
    elif genders[i] == '0' and weights[i] > 110: # чтобы не было много мужских ронни колеманов
        body_fat_percents.append(np.random.uniform(25, 35))
    else:
        body_fat_percents.append(np.random.uniform(13, 35))
# Генерация целевого веса на основе и веса
target_weights = []
for i in range(data_size):
    if weights[i] <= 70:
        target_weights.append(weights[i] * (1 - np.random.uniform(0.03, 0.18)))
    elif 71 <= weights[i] <= 100:
        target_weights.append(weights[i] * (1 - np.random.uniform(0.05, 0.25)))
    elif 101 <= weights[i] <= 140:
        target_weights.append(weights[i] * (1 - np.random.uniform(0.05, 0.30)))
    else:
        target_weights.append(weights[i] * (1 - np.random.uniform(0.03, 0.20)))
# Дефицит калорий
calorie_deficits = np.random.randint(0, 1000, data_size)

# Рассчитываем количество дней для похудения
days_to_lose_weight = [
    calculate_days_to_lose_weight(weights[i], heights[i], body_fat_percents[i], calorie_deficits[i], target_weights[i], genders[i])
    for i in range(data_size)
]

# Создаем DataFrame и округляем "Days to Lose Weight" и "Target Weight"
df_new = pd.DataFrame({
    "Height": heights,
    "Weight": weights,
    "Body Fat %": np.round(body_fat_percents, 1),
    "Calorie Deficit": calorie_deficits,
    "Target Weight": np.round(target_weights).astype(int),
    "Gender": genders,
    "Days to Lose Weight": np.round(days_to_lose_weight).astype(int)
})

# Разделение на признаки и целевую переменную
X = df_new.drop("Days to Lose Weight", axis=1)
y = df_new["Days to Lose Weight"]

# Нормализация числовых признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на тренировочные и тестовые наборы
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Создание нейронной сети
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),  # Регуляризация
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1)  # Один выходной нейрон для регрессии
])

# Компиляция модели
model.compile(optimizer='adam', loss='mae', metrics=['mae'])

# Обучение модели
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])


# Оценка модели
y_pred = model.predict(X_test).flatten()
mae = mean_absolute_error(y_test, y_pred)
print(f"Neural Network: MAE = {mae:.2f}")
# Генерация данных для случайных людей, аналогичных основному датасету
def generate_random_people(num_people=10):
    genders = np.random.choice([0, 1], num_people)  # 0: Male, 1: Female
    heights = [np.random.randint(160, 191) if g == 0 else np.random.randint(150, 190) for g in genders]
    weights = [np.random.randint(60, 140) if g == 0 else np.random.randint(50, 120) for g in genders]
    body_fat_percents = [
        np.random.uniform(13, 25) if weights[i] <= 70 else np.random.uniform(25, 35)
        if weights[i] > 100 else np.random.uniform(13, 35)
        for i in range(num_people)
    ]
    target_weights = [
        weights[i] * (1 - np.random.uniform(0.05, 0.25)) for i in range(num_people)
    ]
    calorie_deficits = np.random.randint(200, 1000, num_people)
    ages = np.random.randint(16, 60, num_people)

    # Рассчет дней для похудения
    days_to_lose_weight = [
        calculate_days_to_lose_weight(weights[i], heights[i], body_fat_percents[i], calorie_deficits[i], target_weights[i], ages[i], genders[i])
        for i in range(num_people)
    ]

    data = pd.DataFrame({
        'Height': heights,
        'Weight': weights,
        'Body Fat %': np.round(body_fat_percents, 1),
        'Calorie Deficit': calorie_deficits,
        'Target Weight': np.round(target_weights).astype(int),
        'Gender': genders,
        'Days to Lose Weight': np.round(days_to_lose_weight).astype(int)
    })
    return data

# Генерация данных
random_people = generate_random_people(20)

# Стандартизация данных (используется обученный scaler)
# Предполагается, что scaler уже обучен на обучающем наборе данных
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Загрузить обученный scaler и стандартизировать данные
scaled_data = scaler.fit_transform(random_people.drop(columns=['Days to Lose Weight']))

# Предсказания для случайных людей
predictions = model.predict(scaled_data)
random_people['Predicted Days to Lose Weight'] = predictions
# Вывод результатов
print(random_people)
# Преобразование данных для предсказаний
test_data = pd.DataFrame({
    'Height': [175],
    'Weight': [75],
    'Body Fat %': [20],
    'Calorie Deficit': [600],
    'Target Weight': [68],
    'Gender': [0]  # Пример данных для женщины
})
# Нормализация данных
scaled_test_data = scaler.transform(test_data)
# Предсказание
prediction = model.predict(scaled_test_data)
print("Predicted days to lose weight:", prediction)

model.save("weight_loss_model.keras")
import joblib
joblib.dump(scaler, 'scaler.pkl')

