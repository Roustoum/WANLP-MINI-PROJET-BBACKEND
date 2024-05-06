#coding:UTF-8

from transformers import BertForSequenceClassification,AutoTokenizer,Trainer
import torch

class CoherenceDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]) if self.labels else None
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


DarijaBert_model = BertForSequenceClassification.from_pretrained('./CohereZiClass')
DarijaBERT_tokenizer = AutoTokenizer.from_pretrained('./CohereZiClass')
trainer = Trainer(model=DarijaBert_model)
device = torch.device("cpu")
DarijaBert_model.to(device)




test_texts = ['n5abiw pizza fel ttttt']

test_encodings = DarijaBERT_tokenizer(test_texts, truncation=True, padding=True, max_length=256)

test_dataset = CoherenceDataset(test_encodings, labels=None)


predictions = trainer.predict(test_dataset)

probabilities = torch.nn.functional.softmax(torch.from_numpy(predictions.predictions), dim=-1)

predicted_labels = torch.argmax(probabilities, dim=1)


print(predicted_labels.tolist())