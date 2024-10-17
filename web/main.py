from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import random
import string
from fastapi import Request

# Создание экземпляра FastAPI
app = FastAPI()

# Настройка статических файлов и шаблонов
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def generate_api_key(length=32):
	"""
	Генерирует случайный API-ключ заданной длины.

	:param length: Длина API-ключа (по умолчанию 32 символа).
	:return: Сгенерированный API-ключ.
	"""
	letters_and_digits = string.ascii_letters + string.digits
	return ''.join(random.choice(letters_and_digits) for i in range(length))


@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
	"""
	Возвращает HTML-форму для отправки текста и файла.

	:return: HTML-контент с формой.
	"""
	return templates.TemplateResponse("index.html", {"request": request})


@app.post("/submit")
async def handle_form(text_input: str = Form(...), file: UploadFile = File(None)):
	"""
	Обрабатывает POST-запрос с текстом и файлом.

	:param text_input: Текст, отправленный пользователем.
	:param file: Загруженный файл (опционально).
	:return: JSON-ответ с сообщением и текстом.
	"""
	if file and file.filename.endswith('.csv'):
		return {"message": "Текст и файл получены", "text": text_input}
	else:
		return {"message": "Текст получен без файла", "text": text_input}


@app.get("/generate-api")
async def generate_api():
	"""
	Генерирует и возвращает новый API-ключ.

	:return: JSON-ответ со сгенерированным API-ключом.
	"""
	api_key = generate_api_key()
	return JSONResponse(content={"api_key": api_key})


if __name__ == "__main__":
	"""
	Запуск сервера с использованием Uvicorn.
	"""
	import uvicorn

	uvicorn.run(app, host="127.0.0.1", port=8000)
