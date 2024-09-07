import requests

# Базовый URL для вашего API
base_url = "http://localhost:8000"


def test_review_from_list():
    data = {
        "platform_key": "ebfbwiebflbsvjhbjehf",  # Замените на действительный ключ платформы
        "course_key": "t0ZH",
        "review": ['Преподователь топ, теория тоже. Практика просто офигенная']
    }
    response = requests.post(f"{base_url}/predict/text", data=data)
    print("POST /course:", response.json())


def test_review_from_file():
    file_path = 'test_reviews.csv'

    with open(file_path, 'rb') as file:
        files = {'file': (file_path, file, 'text/csv')}
        data = {
            "platform_key": "ebfbwiebflbsvjhbjehf",  # Замените на действительный ключ платформы
            "course_key": "t0ZH"
        }
        response = requests.post('http://localhost:8000/predict/file', files=files, data=data)

        # Проверка успешности запроса
        if response.status_code == 200:
            print('Success:', response.json())
        else:
            print('Failed:', response.status_code, response.text)


# Тестирование эндпоинта /statistics
def test_statistics():
    params = {
        "platform_key": "ebfbwiebflbsvjhbjehf",  # Замените на действительный ключ платформы
        "course_key": "t0ZH"  # Замените на действительный ключ курса
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


def test_del_course():
    data = {
        "platform_key": "ebfbwiebflbsvjhbjehf",  # Замените на действительный ключ платформы
        "course_key": "t0ZH"  # Замените на действительный ключ курса
    }
    response = requests.delete(f"{base_url}/course", json=data)
    print("POST /course:", response.json())


def test_add_user():
    data = {
        "secret_key": "23b301fe34fecfb712ee62fd33069686",  # Замените на действительный ключ
        "tg_key": "213212"  # Замените на действительный ключ
    }
    response = requests.post(f"{base_url}/add_user", json=data)
    print("POST /add_user:", response.json())

def test_get_user():
    params = {
        "secret_key": "23b301fe34fecfb712ee62fd33069686",  # Замените на действительный ключ
        "tg_id": "213212"  # Замените на действительный ключ
    }
    response = requests.get(f"{base_url}/get_user", params=params)
    print("GET /get_user:", response.json())
