import requests

# Базовый URL для вашего API
base_url = "http://127.0.0.1:42"


# Тестирование эндпоинта /statistics
def test_statistics():
    params = {
        "platform_key": "ebfbwiebflbsvjhbjehf",  # Замените на действительный ключ платформы
        "course_key": "ltEd"  # Замените на действительный ключ курса
    }
    response = requests.get(f"{base_url}/statistics", params=params)
    print("GET /statistics:", response.json())


# Тестирование эндпоинта /course для добавления курса
def test_add_course():
    data = {
        "platform_key": "ebfbwiebflbsvjhbjehf"  # Замените на действительный ключ платформы
    }
    response = requests.post(f"{base_url}/course", json=data)
    print("POST /course:", response.json())


# Тестирование эндпоинта /course для удаления курса
def test_delete_course():
    data = {
        "platform_key": "ebfbwiebfbsvjhbjehf",  # Замените на действительный ключ платформы
        "course_key": "ltEd"  # Замените на действительный ключ курса
    }
    response = requests.delete(f"{base_url}/course", json=data)
    print("DELETE /course:", response.json())


# Выполнение тестов
if __name__ == "__main__":
    test_delete_course()
    test_statistics()
