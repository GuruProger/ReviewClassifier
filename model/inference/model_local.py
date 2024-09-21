import torch
import pandas as pd
from transformers import AutoTokenizer
from core.model import get_model
from core.data_loader import ReviewPredictor

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


def predict_text(text: str) -> pd.DataFrame:
	df = pd.DataFrame({"Reviews": [text]})
	predictor = ReviewPredictor(df, tokenizer, MAX_LEN)
	predicted_df = predictor.predict(model, device, TOPIC_KEYWORDS, batch_size=1)
	return predicted_df


def predict_file(file_path: str) -> pd.DataFrame:
	df = pd.read_csv(file_path)
	# Проверяем, является ли первый столбец числовым индексом
	if df.index.is_numeric():
		# Если первый столбец - числовым индексом, используем его как индекс
		df.index.name = 'id'
	else:
		# Если первый столбец не является числовым индексом, продолжаем работу без изменений
		pass

	print(df.columns[0])
	try:
		df = df.drop(['практика', 'теория', 'преподаватель', 'технологии', 'актуальность'], axis=1)
	except:
		pass
	predictor = ReviewPredictor(df, tokenizer, MAX_LEN)
	predicted_df = predictor.predict(model, device, TOPIC_KEYWORDS, batch_size=BATCH_SIZE)
	return predicted_df


if __name__ == '__main__':
	# Пример использования для предсказания текста
	review_text = """Этот курс оставил у меня только положительные впечатления благодаря тому, как были поданы материалы. Ведущий демонстрировал высокий уровень знаний, что помогало глубже понять сложные темы. Каждый новый раздел был структурирован логично, с понятными примерами и практическими заданиями. Особое внимание уделялось важным деталям, что значительно облегчало усвоение информации. Рекомендую этот курс всем."""
	result_df = predict_text(review_text)
	print(result_df.iloc[0])

	# Пример использования для предсказания из файла
	file_path = "../data/test_reviews.csv"
	result_df = predict_file(file_path)
	result_df.to_csv('../data/answer_bert.csv')