    FROM python:3.12

    # Устанавливаем рабочую директорию
    WORKDIR /app

    # Копируем файл с зависимостями
    COPY requements.txt requements.txt

    # Обновляем setuptools перед установкой зависимостей
    RUN pip install --upgrade setuptools

    # Устанавливаем зависимости
    RUN pip install -r requements.txt

    # Устанавливаем разрешения для рабочей директории
    RUN chmod 755 /app

    # Копируем все файлы проекта в контейнер
    COPY . .

    # Запуск приложения
    CMD ["python", "main.py"]