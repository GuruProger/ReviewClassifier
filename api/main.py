from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
import httpx
from database import DatabaseClient

# Инициализация FastAPI приложения
app = FastAPI()


@app.post("/predict/text")
async def review_from_list(
        platform_key: str = Form(...),  # Получение ключа платформы из данных формы
        course_key: str = Form(...), # Получение ключа курса из данных формы
        review: list[str] = Form(...) # Получение отзывов курса из данных формы
):
    # Проверка наличия ключа платформы
    if not platform_key:
        raise HTTPException(status_code=400, detail="Missing platform_key")  # Ошибка, если ключ отсутствует

    with DatabaseClient() as db:
        user = db.get_user_by_key(platform_key)  # Получение данных пользователя по ключу платформы
        if not user:
            raise HTTPException(status_code=404, detail="Invalid platform key")  # Ошибка, если пользователь не найден

    # URL внешнего сервиса для отправки данных
    url = "http://localhost:8001/predict/text"
    data = {"platform_key": platform_key, "course_key": course_key, "review": review}
    try:
        async with httpx.AsyncClient() as client:
            # Отправка POST запроса на внешний сервис с JSON-данными
            response = await client.post(url, data=data)
        response.raise_for_status()  # Проверка, что запрос выполнен успешно
    except httpx.HTTPStatusError as e:
        # Обработка ошибки HTTP статуса, если ответ сервера содержит ошибку
        raise HTTPException(status_code=response.status_code, detail="Unexpected error") from e
    except httpx.RequestError as e:
        # Обработка ошибки запроса (например, сетевой сбой)
        raise HTTPException(status_code=500, detail="Internal server error") from e

    # Возврат успешного результата
    return {"status": "success"}


@app.post("/predict/file")
async def review_from_file(
        file: UploadFile = File(...),  # Получение загруженного файла
        platform_key: str = Form(...),  # Получение ключа платформы из данных формы
        course_key: str = Form(...)  # Получение ключа курса из данных формы
):
    # Проверка типа загружаемого файла
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")  # Ошибка при неправильном типе файла

    # Проверка наличия ключей платформы и курса
    if not platform_key or not course_key:
        raise HTTPException(status_code=400, detail="Missing platform_key or course_key")  # Ошибка при отсутствии ключей

    # Синхронная работа с базой данных
    with DatabaseClient() as db:
        user = db.get_user_by_key(platform_key)  # Проверка пользователя по ключу платформы
        if not user:
            raise HTTPException(status_code=404, detail="Invalid platform key")  # Ошибка при отсутствии пользователя

    # Чтение содержимого загруженного файла
    file_content = await file.read()

    # URL внешнего сервиса для отправки файла
    forward_url = "http://localhost:8001/predict/file"
    files = {"file": (file.filename, file_content, file.content_type)}  # Подготовка файла для отправки
    data = {"platform_key": platform_key, "course_key": course_key}  # Подготовка дополнительных данных для запроса

    try:
        async with httpx.AsyncClient() as client:
            # Отправка POST запроса с файлом и дополнительными данными
            response = await client.post(forward_url, files=files, data=data)
        response.raise_for_status()  # Проверка успешности запроса
    except httpx.HTTPStatusError as e:
        # Обработка ошибки HTTP статуса
        raise HTTPException(status_code=response.status_code, detail="Unexpected error") from e
    except httpx.RequestError as e:
        # Обработка сетевых ошибок
        raise HTTPException(status_code=500, detail="Internal server error") from e

    # Возврат успешного результата
    return {"status": "success"}


@app.get("/statistics")
async def statistics(request: Request):
    # Получение параметров platform_key и course_key из строки запроса
    platform_key = request.query_params.get("platform_key")
    course_key = request.query_params.get("course_key")

    # Проверка наличия ключей платформы и курса
    if not platform_key or not course_key:
        raise HTTPException(status_code=400, detail="Missing platform_key or course_key")  # Ошибка при отсутствии ключей

    # Синхронная работа с базой данных
    with DatabaseClient() as db:
        user = db.get_user_by_key(platform_key)  # Проверка существования пользователя
        if user:
            result = db.get_statistics_for_course(user[0], course_key)  # Получение статистики по курсу
            if result:
                return {"status": "success", "result": result}  # Возврат успешного результата
            else:
                raise HTTPException(status_code=404, detail="Course statistics not found")  # Ошибка, если статистика курса не найдена
        else:
            raise HTTPException(status_code=404, detail="Invalid platform key")  # Ошибка, если пользователь не найден


@app.post("/course")
async def course_add(request: Request):
    data = await request.json()  # Получение данных из запроса
    platform_key = data.get("platform_key")  # Извлечение ключа платформы

    # Проверка наличия ключа платформы
    if not platform_key:
        raise HTTPException(status_code=400, detail="Missing platform_key")  # Ошибка, если ключ отсутствует

    # Синхронная работа с базой данных
    with DatabaseClient() as db:
        user = db.get_user_by_key(platform_key)  # Проверка существования пользователя
        if user:
            result = db.add_course(user[0])  # Добавление курса для пользователя
            if result:
                return {"status": "success", "result": result}  # Возврат успешного результата
            else:
                raise HTTPException(status_code=500, detail="Failed to add course")  # Ошибка при добавлении курса
        else:
            raise HTTPException(status_code=404, detail="Invalid platform key")  # Ошибка, если пользователь не найден


@app.delete("/course")
async def course_delete(request: Request):
    data = await request.json()  # Получение данных из запроса
    platform_key = data.get("platform_key")  # Извлечение ключа платформы
    course_key = data.get("course_key")  # Извлечение ключа курса

    # Проверка наличия ключей платформы и курса
    if not platform_key or not course_key:
        raise HTTPException(status_code=400, detail="Missing platform_key or course_key")  # Ошибка, если ключи отсутствуют

    # Синхронная работа с базой данных
    with DatabaseClient() as db:
        user = db.get_user_by_key(platform_key)  # Проверка существования пользователя
        if user:
            result = db.del_course(user[0], course_key)  # Удаление курса
            if result == "ok":
                return {"status": "success", "result": result}  # Возврат успешного результата
            else:
                raise HTTPException(status_code=500, detail="Failed to delete course")  # Ошибка при удалении курса
        else:
            raise HTTPException(status_code=404, detail="Invalid platform key")  # Ошибка, если пользователь не найден


@app.post("/save_the_prediction")
async def save_the_prediction(request: Request):
    data = await request.json()  # Получение данных из запроса
    with DatabaseClient() as db:
        user = db.get_user_by_key(data["platform_key"])  # Проверка существования пользователя по ключу платформы
        res = db.update_statistics_for_course(user[0], data["course_key"], data["predictions"])  # Обновление статистики курса
        if res:
            return {"status": "success"}  # Возврат успешного результата
    return {"status": "error"}  # Возврат ошибки при неудачном обновлении


if __name__ == '__main__':
    import uvicorn
    # Запуск приложения FastAPI с помощью uvicorn
    uvicorn.run(app, host="localhost", port=8000)
