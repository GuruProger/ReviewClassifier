import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class ReviewPredictor(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.reviews = df['Reviews'].to_numpy()
        self.tokenizer = tokenizer
        self.max_len = max_len

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

    def create_data_loader(self, batch_size):
        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=1
        )

    def predict(self, model, device, topic_keywords, batch_size=64):
        data_loader = self.create_data_loader(batch_size)
        predictions = []
        model.eval()

        with torch.no_grad():
            for data in data_loader:
                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = (outputs > 0.5).to(torch.int).cpu().numpy()
                predictions.extend(preds)

        predictions_df = pd.DataFrame(predictions, columns=topic_keywords)
        return pd.concat([self.df, predictions_df], axis=1)
