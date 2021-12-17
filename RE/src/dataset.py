import config
import torch
import transformers

class REDataset:
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        inputs = config.TOKENIZER.encode_plus(text, None, add_special_tokens=True, max_length=config.MAX_LEN, padding="max_length", truncation=True)

        return {
            "ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(inputs["token_type_ids"], dtype=torch.long),
            "targets": torch.tensor(label, dtype=torch.long),
        }

