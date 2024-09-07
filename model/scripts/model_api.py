import torch
from transformers import AutoTokenizer
from model import get_model
import io
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
import httpx
import pandas as pd

URL = "http://localhost:8000"

# Конфигурация модели и токенизатора
# Имя предварительно обученной модели и параметры
PRE_TRAINED_MODEL_NAME = 'DeepPavlov/rubert-base-cased'
MAX_LEN = 250  # Максимальная длина токенизированного текста
BATCH_SIZE = 64  # Размер пакета для предсказаний
TOPIC_KEYWORDS = ['практика', 'теория', 'преподаватель', 'технологии', 'актуальность']  # Список ключевых слов,
# связанных с темами

# Инициализация токенизатора и модели
# Создание токенизатора на основе предобученной модели
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, clean_up_tokenization_spaces=True)

# Получение модели с заданным количеством классов (в данном случае, по числу ключевых слов)
model = get_model(n_classes=len(TOPIC_KEYWORDS))

# Определение устройства для вычислений (GPU или CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Перемещение модели на выбранное устройство
model.to(device)

# Установка модели в режим оценки (инференса)
model.eval()


# Функция для получения предсказания для одного или нескольких отзывов
def predict(reviews):
    predictions = []
    for review_text in reviews:
        # Токенизация текста отзыва
        encoded_review = tokenizer.encode_plus(
            review_text,
            max_length=MAX_LEN,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        # Получение input_ids и attention_mask и перенос их на устройство (GPU/CPU)
        input_ids = encoded_review['input_ids'].to(device)
        attention_mask = encoded_review['attention_mask'].to(device)

        # Получение предсказания модели, отключив вычисление градиентов для ускорения
        with torch.no_grad():
            output = model(input_ids, attention_mask)

        # Очистка памяти GPU
        torch.cuda.empty_cache()

        # Преобразование предсказаний в бинарный формат (0 или 1) и добавление в список предсказаний
        prediction = (output > 0.5).to(torch.int).tolist()[0]
        predictions.append(prediction)
    return predictions


def process_predictions(reviews, platform_key, course_key):
    # Здесь выполняется обработка предсказаний
    predictions = predict(reviews)  # Пример предсказаний, замените на ваш алгоритм
    results = {
        "platform_key": platform_key,
        "course_key": course_key,
        "predictions": [sum(x) for x in zip(*predictions)]
    }
    try:
        response = httpx.post(f"{URL}/save_the_prediction", data=results)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        # Обработка ошибки отправки
        print(f"Failed to send results: {e.response.text}")


# Инициализация FastAPI приложения
app = FastAPI()


# Обработчик POST запроса для предсказания по текстам отзывов
@app.post("/predict/text")
async def process_review(
        background_tasks: BackgroundTasks,
        platform_key: str = Form(...),
        course_key: str = Form(...),
        review: list[str] = Form(...)
):
    # Постановка задачи в фоновую очередь
    background_tasks.add_task(process_predictions, review, platform_key, course_key)

    # Возвращаем ответ клиенту
    return {"status": "Task accepted and is being processed."}


# Обработчик POST запроса для предсказания на основе загруженного CSV файла
@app.post("/predict/file")
async def process_csv(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        platform_key: str = Form(...),
        course_key: str = Form(...)
):
    # Чтение и декодирование содержимого CSV файла
    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode('utf-8')))

    # Проверка наличия столбца 'review' в CSV файле
    if 'review' not in df.columns:
        raise HTTPException(status_code=400, detail="CSV file must contain a 'review' column.")

    # Преобразование столбца отзывов в список
    reviews = df['review'].tolist()

    # Постановка задачи в фоновую очередь
    background_tasks.add_task(process_predictions, reviews, platform_key, course_key)

    # Возвращаем ответ клиенту
    return {"status": "Task accepted and is being processed."}


# Запуск приложения при запуске скрипта напрямую
if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="localhost", port=8001)  # Запуск Uvicorn сервера на порту 8001
