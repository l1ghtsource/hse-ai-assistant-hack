# Используем базовый образ Python
FROM python:3.11-slim

# Установим зависимости системы (если необходимы, например для pandas и других)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /usr/src/app

# Копируем файлы проекта
COPY ./src ./src
COPY ./data ./data
COPY ./notebooks ./notebooks
COPY ./src/requirements.txt .

# Установим зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Выполним команду для обучения модели (если нужно)
# Для обучения установи переменную окружения TRAIN=true в docker-compose.yaml
ARG TRAIN=false

# Если train=true, запускаем обучение, если false, пропускаем
RUN if [ "$TRAIN" = "true" ]; then \
        python src/train.py; \
    fi

# Запускаем инференс и конвертацию
CMD ["bash", "-c", "python src/inference.py && python src/convert.py"]
