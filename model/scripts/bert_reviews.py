import os
from torch import nn
import numpy as np
from transformers import AutoModel
from transformers import AutoTokenizer
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


PRE_TRAINED_MODEL_NAME = 'DeepPavlov/rubert-base-cased'
MAX_LEN = 250


tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, clean_up_tokenization_spaces=True)


class ReviewsClassifier(nn.Module):
	def __init__(self, n_classes):
		super().__init__()
		self.bert = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
		self.drop = nn.Dropout(p=0.3)

		# Создаем отдельный линейный слой для каждого класса
		self.classifiers = nn.ModuleList([
			nn.Linear(self.bert.config.hidden_size, 1) for _ in range(n_classes)
		])

	def forward(self, input_ids, attention_mask):
		_, pooled_output = self.bert(
			input_ids=input_ids,
			attention_mask=attention_mask,
			return_dict=False)

		output = self.drop(pooled_output)

		# Применяем каждый линейный слой к одному и тому же входу, но независимо
		outputs = [torch.sigmoid(classifier(output)) for classifier in self.classifiers]

		# Соединяем все предсказания в один тензор
		return torch.cat(outputs, dim=1)


n_classes = 5
EPOCHS = 2
topic_keywords = ['практика', 'теория', 'преподаватель', 'технологии', 'актуальность']


# Получаем путь к текущему файлу и строим путь к нужному файлу
path_weights = Path(__file__).resolve().parent / 'weights' / 'saved_weights.pt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ReviewsClassifier(n_classes)
model = model.to(device)
model.eval()

if os.path.exists(
		path_weights):  # Во время разработки нужно игнорировать файл с весами, запуск готового приложения будет не из этого файла
	print("Файл с весами найден. Загружаем веса вместо обучения.")
	model.load_state_dict(torch.load(path_weights, weights_only=False))

# Примерные значения, которые вы должны уже иметь из своего кода
MAX_LEN = 250




class ReviewPredictor(Dataset):
	def __init__(self, df, tokenizer, max_len, batch_size):
		self.reviews = df['Reviews'].to_numpy()
		self.tokenizer = tokenizer
		self.max_len = max_len
		self.batch_size = batch_size
		self.data_loader = self.create_data_loader()

	def __len__(self):
		return len(self.reviews)

	def __getitem__(self, item):
		review = str(self.reviews[item])
		encoding = self.tokenizer.encode_plus(
			review,
			max_length=self.max_len,
			add_special_tokens=True,
			return_token_type_ids=False,
			padding='max_length',
			return_attention_mask=True,
			return_tensors='pt',
			truncation=True
		)
		return {
			'input_ids': encoding['input_ids'].flatten(),
			'attention_mask': encoding['attention_mask'].flatten()
		}

	def create_data_loader(self):
		return DataLoader(
			self,
			batch_size=self.batch_size,
			num_workers=1
		)

	def predict(self, model, device, topic_keywords):
		predictions = []
		model.eval()

		with torch.no_grad():
			for data in self.data_loader:
				input_ids = data['input_ids'].to(device)
				attention_mask = data['attention_mask'].to(device)

				outputs = model(input_ids=input_ids, attention_mask=attention_mask)
				preds = (outputs > 0.5).to(torch.int).cpu().numpy()
				predictions.extend(preds)

		predictions = np.array(predictions)
		df[topic_keywords] = predictions
		return df


if __name__ == '__main__':
	# Загружаем CSV

	df = pd.read_csv('../data/test_reviews.csv')
	predictor = ReviewPredictor(df, tokenizer, MAX_LEN, batch_size=64)

	# Предсказываем
	predicted_df = predictor.predict(model, device, topic_keywords)
