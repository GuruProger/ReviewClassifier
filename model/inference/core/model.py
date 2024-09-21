import torch
from torch import nn
from transformers import AutoModel
from pathlib import Path
from .settings import PRE_TRAINED_MODEL_NAME

path_weights = Path(__file__).resolve().parent.parent.parent / 'weights' / 'saved_weights.pt'


class ReviewsClassifier(nn.Module):
	def __init__(self, n_classes):
		super().__init__()
		self.bert = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
		self.drop = nn.Dropout(p=0.3)
		self.classifiers = nn.ModuleList([
			nn.Linear(self.bert.config.hidden_size, 1) for _ in range(n_classes)
		])

	def forward(self, input_ids, attention_mask):
		_, pooled_output = self.bert(
			input_ids=input_ids,
			attention_mask=attention_mask,
			return_dict=False
		)
		output = self.drop(pooled_output)
		outputs = [torch.sigmoid(classifier(output)) for classifier in self.classifiers]
		return torch.cat(outputs, dim=1)


_model_instance = None


def get_model(n_classes=5, device=None):
	global _model_instance
	if _model_instance is None:
		device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		model = ReviewsClassifier(n_classes)
		model = model.to(device)

		# Проверка пути с помощью assert
		assert path_weights.exists(), f"Weight file not found at: {path_weights}"
		print("Loading model weights from:", path_weights)
		model.load_state_dict(torch.load(path_weights, map_location=device))

		_model_instance = model

	return _model_instance
