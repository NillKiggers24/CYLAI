# Используем официальный образ Python
FROM python:3.9-slim
# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файлы проекта в контейнер
COPY . /app
# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn
RUN chmod 666 /app/instance/Weightloss.db
# Указываем порт, на котором будет работать приложение
EXPOSE 5000

# Команда запуска приложения
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]