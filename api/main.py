# Импорт необходимых библиотек и модулей
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, Query
from pydantic import BaseModel, Field
import httpx
from typing import List, Dict, Optional
from database import DatabaseClient

# Константы
SECRET_KEY = "23b301fe34fecfb712ee62fd33069686"
URL = "http://localhost:8001"

# Инициализация приложения FastAPI
app = FastAPI()


# Эндпоинт для предсказания по текстовым отзывам
@app.post("/predict/text", summary="Предсказание по текстовым отзывам",
		  description="Принимает ключ платформы, ключ курса и текстовые отзывы. Отправляет данные на внешний сервис для получения предсказания и возвращает результат.")
async def review_from_list(
		platform_key: str = Form(..., description="Ключ платформы пользователя"),
		course_key: str = Form(..., description="Ключ курса"),
		review: List[str] = Form(..., description="Список текстовых отзывов")
):
	# Проверка наличия ключа платформы
	if not platform_key:
		raise HTTPException(status_code=400, detail="Missing platform_key")

	# Проверка действительности ключа платформы в базе данных
	with DatabaseClient() as db:
		user = db.get_user_by_key(platform_key)
		if not user:
			raise HTTPException(status_code=404, detail="Invalid platform key")

	data = {"platform_key": platform_key, "course_key": course_key, "review": review}

	try:
		# Отправка данных на внешний сервис для получения предсказания
		async with httpx.AsyncClient() as client:
			response = await client.post(f"{URL}/predict/text", data=data)
		response.raise_for_status()
	except httpx.HTTPStatusError as e:
		raise HTTPException(status_code=response.status_code, detail="Unexpected error") from e
	except httpx.RequestError as e:
		raise HTTPException(status_code=500, detail="Internal server error") from e

	return {"status": "success"}


# Эндпоинт для предсказания по загруженному файлу
@app.post("/predict/file", summary="Предсказание по загруженному файлу", description="Принимает загруженный CSV файл, ключ платформы и ключ курса. Отправляет файл на внешний сервис для получения предсказания и возвращает результат.")
async def review_from_file(
		file: UploadFile = File(..., description="Загруженный CSV файл с отзывами"),
		platform_key: str = Form(..., description="Ключ платформы пользователя"),
		course_key: str = Form(..., description="Ключ курса")
):
	# Проверка типа загруженного файла
	if file.content_type != 'text/csv':
		raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")

	# Проверка наличия ключей
	if not platform_key or not course_key:
		raise HTTPException(status_code=400, detail="Missing platform_key or course_key")

	# Проверка действительности ключа платформы в базе данных
	with DatabaseClient() as db:
		user = db.get_user_by_key(platform_key)
		if not user:
			raise HTTPException(status_code=404, detail="Invalid platform key")

	# Чтение содержимого файла
	file_content = await file.read()
	files = {"file": (file.filename, file_content, file.content_type)}
	data = {"platform_key": platform_key, "course_key": course_key}

	try:
		# Отправка файла на внешний сервис для получения предсказания
		async with httpx.AsyncClient() as client:
			response = await client.post(f"{URL}/predict/file", files=files, data=data)
		response.raise_for_status()
	except httpx.HTTPStatusError as e:
		raise HTTPException(status_code=response.status_code, detail="Unexpected error") from e
	except httpx.RequestError as e:
		raise HTTPException(status_code=500, detail="Internal server error") from e

	return {"status": "success"}


# Эндпоинт для получения статистики по курсу
@app.get("/statistics", summary="Получение статистики по курсу",
		 description="Возвращает статистику по курсу для заданных ключей платформы и курса.")
async def statistics(
		platform_key: str = Query(..., description="Ключ платформы пользователя"),
		course_key: str = Query(..., description="Ключ курса")
):
	# Проверка наличия ключей
	if not platform_key or not course_key:
		raise HTTPException(status_code=400, detail="Missing platform_key or course_key")

	# Получение статистики по курсу из базы данных
	with DatabaseClient() as db:
		user = db.get_user_by_key(platform_key)
		if user:
			result = db.get_statistics_for_course(user[0], course_key)
			if result:
				return {"status": "success", "result": result}
			else:
				raise HTTPException(status_code=404, detail="Course statistics not found")
		else:
			raise HTTPException(status_code=404, detail="Invalid platform key")


# Эндпоинт для добавления нового курса
@app.post("/course", summary="Добавление курса",
		  description="Добавляет новый курс для пользователя по ключу платформы.")
async def course_add(
		platform_key: str = Form(..., description="Ключ платформы пользователя")
):
	# Проверка наличия ключа платформы
	if not platform_key:
		raise HTTPException(status_code=400, detail="Missing platform_key")

	# Добавление нового курса в базу данных
	with DatabaseClient() as db:
		user = db.get_user_by_key(platform_key)
		if user:
			result = db.add_course(user[0])
			if result:
				return {"status": "success", "result": result}
			else:
				raise HTTPException(status_code=500, detail="Failed to add course")
		else:
			raise HTTPException(status_code=404, detail="Invalid platform key")


# Эндпоинт для удаления курса
@app.delete("/course", summary="Удаление курса", description="Удаляет курс по ключу платформы и курса.")
async def course_delete(
		platform_key: str = Form(..., description="Ключ платформы пользователя"),
		course_key: str = Form(..., description="Ключ курса")
):
	# Проверка наличия ключей
	if not platform_key or not course_key:
		raise HTTPException(status_code=400, detail="Missing platform_key or course_key")

	# Удаление курса из базы данных
	with DatabaseClient() as db:
		user = db.get_user_by_key(platform_key)
		if user:
			result = db.del_course(user[0], course_key)
			if result == "ok":
				return {"status": "success", "result": result}
			else:
				raise HTTPException(status_code=500, detail="Failed to delete course")
		else:
			raise HTTPException(status_code=404, detail="Invalid platform key")


# Эндпоинт для сохранения предсказаний
@app.post("/save_the_prediction", summary="Сохранение предсказаний",
		  description="Обновляет статистику курса в базе данных с новыми предсказаниями.")
async def save_the_prediction(
		platform_key: str = Form(..., description="Ключ платформы пользователя"),
		course_key: str = Form(..., description="Ключ курса"),
		predictions: List = Form(..., description="Предсказания для курса")
):
	print(predictions, type(predictions))
	# Обновление статистики курса с новыми предсказаниями
	with DatabaseClient() as db:
		user = db.get_user_by_key(platform_key)
		res = db.update_statistics_for_course(user[0], course_key, predictions)
		if res:
			return {"status": "success"}
	return {"status": "error"}


# Эндпоинт для добавления нового пользователя
@app.post("/add_user", summary="Добавление нового пользователя",
		  description="Добавляет нового пользователя по ключу платформы и возвращает ключ пользователя. Требуется секретный ключ для авторизации.")
async def add_user(
		secret_key: str = Form(..., description="Секретный ключ для авторизации"),
		tg_key: str = Form(..., description="ТГ ключ пользователя")
):
	# Проверка секретного ключа
	if secret_key != SECRET_KEY:
		return {"status": "error: wrong key"}

	# Добавление нового пользователя в базу данных
	with DatabaseClient() as db:
		user_key = db.add_user(tg_key)
		if user_key:
			return {"status": "success", "key": user_key}

	return {"status": "error"}


# Эндпоинт для получения данных пользователя
@app.get("/get_user", summary="Получение данных пользователя",
		 description="Возвращает данные пользователя по TG ID при наличии действительного секретного ключа.")
async def get_user(
		secret_key: str = Query(..., description="Секретный ключ для авторизации"),
		tg_id: str = Query(..., description="TG ID пользователя")
):
	# Проверка секретного ключа
	if secret_key != SECRET_KEY:
		return {"status": "error: wrong key"}

	# Получение данных пользователя из базы данных
	with DatabaseClient() as db:
		user = db.get_user_by_tg(tg_id)
		if user:
			return {"status": "success", "user": user}
		else:
			raise HTTPException(status_code=404, detail="Invalid tg_id")


# Запуск приложения на локальном сервере
if __name__ == '__main__':
	import uvicorn

	uvicorn.run(app, host="localhost", port=8000)
