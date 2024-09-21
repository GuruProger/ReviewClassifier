import requests
import io

API_URL = "http://localhost:8001"


def predict_text(text: str):
	endpoint = f"{API_URL}/predict/text"
	response = None

	try:
		payload = {"text": text}
		response = requests.post(endpoint, json=payload)
		response.raise_for_status()  # Проверка на успешный статус-код (200-299)
		return response.json()

	except requests.exceptions.HTTPError as http_err:
		error_code = response.status_code if response else None
		return {"error_code": error_code, "error_message": str(http_err)}

	except Exception as err:
		return {"error_code": None, "error_message": str(err)}


def predict_file(file):
	endpoint = f"{API_URL}/predict/file"
	response = None
	try:
		response = requests.post(endpoint, files={'file': file})
		response.raise_for_status()  # Проверка на успешный статус-код (200-299)

		return io.BytesIO(response.content)
	except requests.exceptions.HTTPError as http_err:
		error_code = response.status_code if response else None
		return {"error_code": error_code, "error_message": str(http_err)}

	except Exception as err:
		return {"error_code": None, "error_message": str(err)}
