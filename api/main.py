from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
import httpx
from database import DatabaseClient

# Инициализация FastAPI приложения
app = FastAPI()


@app.post("/predict/text")
async def review_from_list(request: Request):
    # Получение данных JSON из тела запроса
    data = await request.json()
    platform_key = data.get("platform_key")

    # Проверка наличия ключа платформы
    if not platform_key:
        raise HTTPException(status_code=400, detail="Missing platform_key")

    with DatabaseClient() as db:
        # Проверка существования пользователя по ключу платформы
        user = db.get_user_by_key(platform_key)
        if not user:
            raise HTTPException(status_code=404, detail="Invalid platform key")

    url = "http://localhost:52"
    try:
        async with httpx.AsyncClient() as client:
            # Отправка POST запроса с JSON-данными
            response = await client.post(url, json=data)
        response.raise_for_status()  # Проверка успешности запроса
    except httpx.HTTPStatusError as e:
        # Обработка ошибок ответа
        raise HTTPException(status_code=response.status_code, detail="Unexpected error") from e
    except httpx.RequestError as e:
        # Обработка ошибок запроса
        raise HTTPException(status_code=500, detail="Internal server error") from e

    return {"status": "success"}


@app.post("/predict/file")
async def review_from_file(
        file: UploadFile = File(...),
        platform_key: str = Form(...),
        course_key: str = Form(...)
):
    # Проверка типа загружаемого файла
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")

    # Проверка наличия ключей платформы и курса
    if not platform_key or not course_key:
        raise HTTPException(status_code=400, detail="Missing platform_key or course_key")

    with DatabaseClient() as db:
        # Проверка существования пользователя по ключу платформы
        user = db.get_user_by_key(platform_key)
        if not user:
            raise HTTPException(status_code=404, detail="Invalid platform key")

    # Чтение содержимого загруженного файла
    file_content = await file.read()

    forward_url = "http://localhost:52/predict/file"
    files = {"file": (file.filename, file_content, file.content_type)}
    data = {"platform_key": platform_key, "course_key": course_key}

    try:
        async with httpx.AsyncClient() as client:
            # Отправка POST запроса с файлом и данными
            response = await client.post(forward_url, files=files, data=data)
        response.raise_for_status()  # Проверка успешности запроса
    except httpx.HTTPStatusError as e:
        # Обработка ошибок ответа
        raise HTTPException(status_code=response.status_code, detail="Unexpected error") from e
    except httpx.RequestError as e:
        # Обработка ошибок запроса
        raise HTTPException(status_code=500, detail="Internal server error") from e

    return {"status": "success"}


@app.get("/statistics")
async def statistics(request: Request):
    # Получение параметров из строки запроса
    platform_key = request.query_params.get("platform_key")
    course_key = request.query_params.get("course_key")

    # Проверка наличия ключей платформы и курса
    if not platform_key or not course_key:
        raise HTTPException(status_code=400, detail="Missing platform_key or course_key")

    with DatabaseClient() as db:
        # Проверка существования пользователя по ключу платформы
        user = db.get_user_by_key(platform_key)
        if user:
            # Получение статистики по курсу
            result = db.get_statistics_for_course(user[0], course_key)
            if result:
                return {"status": "success", "result": result}
            else:
                raise HTTPException(status_code=404, detail="Course statistics not found")
        else:
            raise HTTPException(status_code=404, detail="Invalid platform key")


@app.post("/course")
async def course_add(request: Request):
    # Получение данных JSON из тела запроса
    data = await request.json()
    platform_key = data.get("platform_key")

    # Проверка наличия ключа платформы
    if not platform_key:
        raise HTTPException(status_code=400, detail="Missing platform_key")

    with DatabaseClient() as db:
        # Проверка существования пользователя по ключу платформы
        user = db.get_user_by_key(platform_key)
        if user:
            # Добавление нового курса для пользователя
            result = db.add_course(user[0])
            if result:
                return {"status": "success", "result": result}
            else:
                raise HTTPException(status_code=500, detail="Failed to add course")
        else:
            raise HTTPException(status_code=404, detail="Invalid platform key")


@app.delete("/course")
async def course_delete(request: Request):
    # Получение данных JSON из тела запроса
    data = await request.json()
    platform_key = data.get("platform_key")
    course_key = data.get("course_key")

    # Проверка наличия ключей платформы и курса
    if not platform_key or not course_key:
        raise HTTPException(status_code=400, detail="Missing platform_key or course_key")

    with DatabaseClient() as db:
        # Проверка существования пользователя по ключу платформы
        user = db.get_user_by_key(platform_key)
        if user:
            # Удаление курса по ключу курса
            result = db.del_course(user[0], course_key)
            if result == "ok":
                return {"status": "success", "result": result}
            else:
                raise HTTPException(status_code=500, detail="Failed to delete course")
        else:
            raise HTTPException(status_code=404, detail="Invalid platform key")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)