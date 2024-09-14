from dotenv import load_dotenv
import os
import requests
import httpx

load_dotenv()
secret_key = os.getenv("SECRET_KEY")
BASE_URL = "http://localhost:8000"


def add_user(user_id: int | str):
	data = {
		"secret_key": secret_key,
		"tg_id": user_id
	}

	try:
		response = requests.post(f"{BASE_URL}/add_user", data=data)
		response.raise_for_status()
		return response.json()
	except httpx.HTTPStatusError as e:
		# Обработка ошибки отправки
		print(f"Error: {e.response.text}")
		return {'status': 'error'}


def get_user(user_id: int | str):
	params = {
		"secret_key": secret_key,
		"tg_id": user_id
	}
	try:
		response = requests.get(f"{BASE_URL}/get_user", params=params)

		response.raise_for_status()
		return response.json()

	except requests.exceptions.HTTPError as e:
		if e.response.status_code == 404:
			return {'status': 'error', 'message': 'User not found'}
		else:
			return {'status': 'error', 'message': 'HTTP error'}
	except httpx.HTTPStatusError as e:
		return {'status': 'error', 'message': e.response.text}
