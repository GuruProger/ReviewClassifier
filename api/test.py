import requests

# Базовый URL для вашего API
BASE_URL = "http://localhost:8000"
PLATFORM_KEY = "ebfbwiebflbsvjhbjehf"
SECRET_KEY = "23b301fe34fecfb712ee62fd33069686"


def test_review_from_list():
	data = {
		"platform_key": PLATFORM_KEY,  # Замените на действительный ключ платформы
		"course_key": "t0ZH",
		"review": ['Преподователь топ, теория тоже. Практика просто офигенная']
	}
	response = requests.post(f"{BASE_URL}/predict/text", data=data)
	print("POST /predict/text:", response.json())


def test_review_from_file():
	file_path = 'test_reviews.csv'

	with open(file_path, 'rb') as file:
		files = {'file': (file_path, file, 'text/csv')}
		data = {
			"platform_key": PLATFORM_KEY,  # Замените на действительный ключ платформы
			"course_key": "t0ZH"
		}
		response = requests.post(f"{BASE_URL}/predict/file", files=files, data=data)

		# Проверка успешности запроса
		if response.status_code == 200:
			print('Success:', response.json())
		else:
			print('Failed:', response.status_code, response.text)


def test_statistics():
	params = {
		"platform_key": PLATFORM_KEY,  # Замените на действительный ключ платформы
		"course_key": "t0ZH"  # Замените на действительный ключ курса
	}
	response = requests.get(f"{BASE_URL}/statistics", params=params)
	print("GET /statistics:", response.json())


def test_add_course():
	data = {
		"platform_key": PLATFORM_KEY  # Замените на действительный ключ платформы
	}
	response = requests.post(f"{BASE_URL}/course", data=data)
	print("POST /course:", response.json())


def test_del_course():
	data = {
		"platform_key": PLATFORM_KEY,  # Замените на действительный ключ платформы
		"course_key": "ZVPU"  # Замените на действительный ключ курса
	}
	response = requests.delete(f"{BASE_URL}/course", data=data)
	print("DELETE /course:", response.json())


def test_add_user():
	data = {
		"secret_key": SECRET_KEY,  # Замените на действительный ключ
		"tg_id": "654321"  # Замените на действительный ключ
	}
	response = requests.post(f"{BASE_URL}/add_user", data=data)
	print("POST /add_user:", response.json())


def test_get_user():
	params = {
		"secret_key": SECRET_KEY,  # Замените на действительный ключ
		"tg_id": "123456"  # Замените на действительный TG ID
	}
	response = requests.get(f"{BASE_URL}/get_user", params=params)
	print("GET /get_user:", response.json())

def test_get_all_course():
	params = {
		"platform_key": PLATFORM_KEY,  # Замените на действительный ключ
	}
	response = requests.get(f"{BASE_URL}/get_all_course", params=params)
	print("GET /get_user:", response.json())


# Запуск тестов
if __name__ == "__main__":
	test_review_from_list()
	# test_review_from_file()
	test_statistics()
	test_add_course()
	test_del_course()
	test_add_user()
	test_get_user()
