# Используем официальный образ Python
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем модель и код
COPY fine-tuned-ruBert /app/fine-tuned-ruBert
COPY app /app/app

# Открываем порт 7860 для доступа к Gradio
EXPOSE 7860

# Команда запуска приложения
CMD ["python", "app/app.py"]