import torch
import pandas as pd
from transformers import AutoTokenizer
from model import get_model
from data_loader import ReviewPredictor

PRE_TRAINED_MODEL_NAME = 'DeepPavlov/rubert-base-cased'
MAX_LEN = 250
BATCH_SIZE = 64
TOPIC_KEYWORDS = ['практика', 'теория', 'преподаватель', 'технологии', 'актуальность']

tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, clean_up_tokenization_spaces=True)




if __name__ == '__main__':
	"""Пример использования"""

	# Загружаем CSV
	df = pd.read_csv('../data/test_reviews.csv')

	# Инициализируем модель и устройство
	model = get_model(n_classes=len(TOPIC_KEYWORDS))
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# Предсказываем
	predictor = ReviewPredictor(df, tokenizer, MAX_LEN)
	predicted_df = predictor.predict(model, device, TOPIC_KEYWORDS, batch_size=BATCH_SIZE)
	predicted_df.head()
