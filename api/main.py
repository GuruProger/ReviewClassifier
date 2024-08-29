from fastapi import FastAPI, Request, HTTPException
from database import DatabaseClient

# Инициализация FastAPI приложения
app = FastAPI()


# Эндпоинт для получения статистики по курсу
@app.get("/statistics")
async def statistics(request: Request):
    # Извлечение параметров platform_key и course_key из строки запроса
    platform_key = request.query_params.get("platform_key")
    course_key = request.query_params.get("course_key")

    if not platform_key or not course_key:
        raise HTTPException(status_code=400, detail="Missing platform_key or course_key")

    with DatabaseClient() as db:
        # Получение пользователя по ключу платформы из базы данных
        user = db.get_user_by_key(platform_key)

        if user:
            # Если пользователь найден, получаем статистику по курсу
            result = db.get_statistics_for_course(user[0], course_key)

            if result:
                # Возвращаем успешный ответ с результатами
                return {"status": "success", "result": result}
            else:
                raise HTTPException(status_code=404, detail="Course statistics not found")

        # Если пользователь не найден, возвращаем ошибку
        raise HTTPException(status_code=404, detail="Invalid platform key")


# Эндпоинт для добавления нового курса
@app.post("/course")
async def course_add(request: Request):
    # Извлечение данных из тела запроса в формате JSON
    data = await request.json()

    platform_key = data.get("platform_key")

    if not platform_key:
        raise HTTPException(status_code=400, detail="Missing platform_key")

    with DatabaseClient() as db:
        # Получение пользователя по ключу платформы из базы данных
        user = db.get_user_by_key(platform_key)
        if user:
            # Если пользователь найден, добавляем курс для этого пользователя
            result = db.add_course(user[0])

            if result:
                # Возвращаем успешный ответ с результатом
                return {"status": "success", "result": result}
            else:
                raise HTTPException(status_code=500, detail="Failed to add course")

        # Если пользователь не найден, возвращаем ошибку
        raise HTTPException(status_code=404, detail="Invalid platform key")


# Эндпоинт для удаления курса
@app.delete("/course")
async def course_delete(request: Request):
    # Извлечение данных из тела запроса в формате JSON
    data = await request.json()

    platform_key = data.get("platform_key")
    course_key = data.get("course_key")

    if not platform_key or not course_key:
        raise HTTPException(status_code=400, detail="Missing platform_key or course_key")

    with DatabaseClient() as db:
        # Получение пользователя по ключу платформы из базы данных
        user = db.get_user_by_key(platform_key)

        if user:
            # Если пользователь найден, удаляем курс по ключу курса
            result = db.del_course(user[0], course_key)

            if result == "ok":
                # Возвращаем успешный ответ с результатом
                return {"status": "success", "result": result}
            else:
                raise HTTPException(status_code=500, detail="Failed to delete course")

        # Если пользователь не найден, возвращаем ошибку
        raise HTTPException(status_code=404, detail="Invalid platform key")
