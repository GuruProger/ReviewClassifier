import torch
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer
from core.model import get_model
from core.data_loader import ReviewPredictor
from typing import Dict
import io

# Конфигурация модели и токенизатора
from core.settings import PRE_TRAINED_MODEL_NAME

MAX_LEN = 250
BATCH_SIZE = 64
TOPIC_KEYWORDS = ['практика', 'теория', 'преподаватель', 'технологии', 'актуальность']

# Инициализация токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, clean_up_tokenization_spaces=True)
model = get_model(n_classes=len(TOPIC_KEYWORDS))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

app = FastAPI()


class TextPredictionRequest(BaseModel):
	text: str


@app.post("/predict/text")
async def predict_text_endpoint(request: TextPredictionRequest):
	if not request.text:
		raise HTTPException(status_code=400, detail="Text not provided")
	try:
		prediction = predict_text(request.text)
		return prediction
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def predict_text(text: str) -> Dict[str, float]:
	df = pd.DataFrame({"Reviews": [text]})
	predictor = ReviewPredictor(df, tokenizer, MAX_LEN)
	predicted_df = predictor.predict(model, device, TOPIC_KEYWORDS, batch_size=1)
	return predicted_df.iloc[0].to_dict()


@app.post("/predict/file")
async def predict_file_endpoint(file: UploadFile = File(...)):
	if not file:
		raise HTTPException(status_code=400, detail="CSV файл не предоставлен")

	try:
		df = pd.read_csv(file.file)
		try:
			df = df.drop(['практика', 'теория', 'преподаватель', 'технологии', 'актуальность'], axis=1)
		except:
			pass
		predictor = ReviewPredictor(df, tokenizer, MAX_LEN)
		predicted_df: pd.DataFrame = predictor.predict(model, device, TOPIC_KEYWORDS, batch_size=BATCH_SIZE)

		# Преобразуем DataFrame в CSV
		csv_buffer = io.StringIO()
		predicted_df.to_csv(csv_buffer, index=False)
		csv_buffer.seek(0)

		parts = file.filename.rsplit('.csv', 1)
		new_filename = (parts[0] if len(parts) > 1 else file.filename) + '_predictions.csv'
		return StreamingResponse(
			iter([csv_buffer.getvalue()]),
			media_type="text/csv",
			headers={"Content-Disposition": f"attachment; filename={new_filename}"}
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Ошибка при предсказании: {str(e)}")


if __name__ == '__main__':
	import uvicorn

	uvicorn.run(app, host="0.0.0.0", port=8001)
