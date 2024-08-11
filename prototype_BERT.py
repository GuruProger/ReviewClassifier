import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

import torch
import transformers
import torch.nn as nn
from transformers import AutoModel, BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Загружаем датасет
df = pd.read_csv("new.csv", index_col=0)
df['Reviews'] = df['Reviews'].astype(str)

# Используем tqdm для отслеживания прогресса
tqdm.pandas()

# Разделяем данные на обучающую, валидационную и тестовую выборки
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.1, random_state=42)

# Извлекаем текстовые данные и метки классов
train_text = train_df['Reviews']
train_labels = train_df[['Практика', 'Теория', 'Преподаватель', 'Технологии', 'Актуальность']]
val_text = val_df['Reviews']
val_labels = val_df[['Практика', 'Теория', 'Преподаватель', 'Технологии', 'Актуальность']]
test_text = test_df['Reviews']
test_labels = test_df[['Практика', 'Теория', 'Преподаватель', 'Технологии', 'Актуальность']]

# Определяем устройство для выполнения вычислений (CPU или GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загружаем предобученную модель BERT и токенизатор
bert = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased-sentence')
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-sentence')

# Токенизация данных с добавлением паддинга и обрезки
tokens_train = tokenizer.batch_encode_plus(
	train_text.values,
	max_length=50,
	padding='max_length',
	truncation=True
)
tokens_val = tokenizer.batch_encode_plus(
	val_text.values,
	max_length=50,
	padding='max_length',
	truncation=True
)
tokens_test = tokenizer.batch_encode_plus(
	test_text.values,
	max_length=50,
	padding='max_length',
	truncation=True
)

# Преобразование токенов в тензоры
train_seq = torch.tensor(tokens_train['input_ids']).to(device)
train_mask = torch.tensor(tokens_train['attention_mask']).to(device)
train_y = torch.tensor(train_labels.values).to(device)

val_seq = torch.tensor(tokens_val['input_ids']).to(device)
val_mask = torch.tensor(tokens_val['attention_mask']).to(device)
val_y = torch.tensor(val_labels.values).to(device)

test_seq = torch.tensor(tokens_test['input_ids']).to(device)
test_mask = torch.tensor(tokens_test['attention_mask']).to(device)
test_y = torch.tensor(test_labels.values).to(device)

# Задаем размер батча
batch_size = 8

# Создаем DataLoader для обучающей и валидационной выборки
train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_seq, val_mask, val_y)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

# Замораживаем параметры модели BERT, чтобы они не обновлялись при обучении
for param in bert.parameters():
	param.requires_grad = False


# Определяем архитектуру модели на основе BERT
class BERT_Arch(nn.Module):

	def __init__(self, bert):
		super(BERT_Arch, self).__init__()
		self.bert = bert
		self.dropout = nn.Dropout(0.1)
		self.relu = nn.ReLU()
		self.fc1 = nn.Linear(768, 512)
		self.fc2 = nn.Linear(512, 5)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, sent_id, mask):
		# Извлекаем CLS-токен для классификации
		_, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
		x = self.fc1(cls_hs)
		x = self.relu(x)
		x = self.dropout(x)
		x = self.fc2(x)
		x = self.softmax(x)
		return x


# Инициализируем модель и перемещаем на устройство
model = BERT_Arch(bert)
model = model.to(device)

from torch.optim import AdamW

# Определяем оптимизатор
optimizer = AdamW(model.parameters(), lr=1e-3)

from sklearn.utils.class_weight import compute_class_weight

# Рассчитываем веса классов для учета дисбаланса
class_weights = compute_class_weight(
	class_weight="balanced",
	classes=np.unique(train_labels.values),
	y=train_labels.values.reshape(-1)
)

# Преобразуем веса классов в тензор и перемещаем на устройство
weights = torch.tensor(class_weights, dtype=torch.float).to(device)
cross_entropy = nn.CrossEntropyLoss()

# Задаем количество эпох
epochs = 20


# Функция обучения модели
def train():
	model.train()
	total_loss = 0
	total_preds = []

	for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
		batch = [r.to(device) for r in batch]
		sent_id, mask, labels = batch
		labels = labels.float()  # Приводим метки к float для расчета потерь
		model.zero_grad()
		preds = model(sent_id, mask)
		loss = nn.BCEWithLogitsLoss()(preds, labels)  # Используем бинарную кросс-энтропию с логитами
		total_loss += loss.item()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Ограничиваем градиенты для стабильности
		optimizer.step()
		preds = preds.detach().cpu().numpy()
		total_preds.append(preds)

	avg_loss = total_loss / len(train_dataloader)
	total_preds = np.concatenate(total_preds, axis=0)

	return avg_loss, total_preds


# Функция оценки модели
def evaluate():
	model.eval()
	total_loss = 0
	total_preds = []

	for step, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
		batch = [t.to(device) for t in batch]
		sent_id, mask, labels = batch

		with torch.no_grad():
			preds = model(sent_id, mask)
			loss = cross_entropy(preds, labels.float())  # Применяем cross-entropy для оценки
			total_loss += loss.item()
			preds = preds.detach().cpu().numpy()
			total_preds.append(preds)

	avg_loss = total_loss / len(val_dataloader)
	total_preds = np.concatenate(total_preds, axis=0)

	return avg_loss, total_preds


best_valid_loss = float('inf')

# Проверка, существует ли файл с весами
path = 'saved_weights.pt'
if os.path.exists(path):
	print("Файл с весами найден. Загружаем веса вместо обучения.")
	model.load_state_dict(torch.load(path))
else:
	# Если сохраненных весов нет, обучаем модель
	best_valid_loss = float('inf')
	train_losses = []
	valid_losses = []

	for epoch in range(epochs):
		print(f'\n Epoch {epoch + 1} / {epochs}')

		train_loss, _ = train()
		valid_loss, valid_preds = evaluate()

		if valid_loss < best_valid_loss:
			best_valid_loss = valid_loss
			torch.save(model.state_dict(), path)  # Сохраняем лучшие веса модели

		train_losses.append(train_loss)
		valid_losses.append(valid_loss)
		print(f'\nTraining loss: {train_loss:.3f}')
		print(f'Validation loss: {valid_loss:.3f}')

	# Загрузка наилучших сохраненных весов модели
	model.load_state_dict(torch.load(path))

# Пример использования модели на новых данных
new_text = ["спасибо "]
new_tokens = tokenizer.batch_encode_plus(
	new_text,
	max_length=50,
	padding='max_length',
	truncation=True
)

new_seq = torch.tensor(new_tokens['input_ids']).to(device)
new_mask = torch.tensor(new_tokens['attention_mask']).to(device)

with torch.no_grad():
	preds = model(new_seq, new_mask)
	preds = preds.detach().cpu().numpy()

# Применяем пороговое значение для классификации
predictions = (preds > 0.5).astype(int)
print(predictions)
