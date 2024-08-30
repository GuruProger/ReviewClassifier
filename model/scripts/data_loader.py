from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
import torch
import pandas as pd
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ReviewPredictor(Dataset):
	def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizer, max_len: int):
		self.df = df
		self.reviews = df['Reviews'].to_numpy()
		self.tokenizer = tokenizer
		self.max_len = max_len
		self.data_loader = None

	def __len__(self) -> int:
		return len(self.reviews)

	def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
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
			'input_ids': encoding['input_ids'].squeeze(0),
			'attention_mask': encoding['attention_mask'].squeeze(0)
		}

	def create_data_loader(self, batch_size: int) -> DataLoader:
		if self.data_loader is None:
			self.data_loader = DataLoader(
				self,
				batch_size=batch_size,
				num_workers=1,
				shuffle=False,
				pin_memory=True  # Оптимизация для GPU
			)
		return self.data_loader

	def predict(self, model: torch.nn.Module, device: torch.device, topic_keywords: List[str],
				batch_size: int = 64) -> pd.DataFrame:
		data_loader = self.create_data_loader(batch_size)
		predictions = []

		model.to(device)
		model.eval()

		with torch.no_grad():
			for idx, data in enumerate(data_loader):
				input_ids = data['input_ids'].to(device)
				attention_mask = data['attention_mask'].to(device)

				outputs = model(input_ids=input_ids, attention_mask=attention_mask)
				preds = (outputs > 0.5).to(torch.int).cpu().numpy()
				predictions.extend(preds)  # Для бинарной классификации

				logger.info(f'Batch {idx + 1}/{len(data_loader)} processed')

		predictions_df = pd.DataFrame(predictions, columns=topic_keywords)
		return pd.concat([self.df, predictions_df], axis=1)
