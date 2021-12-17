import config
import torch
import torch.nn as nn
import transformers

def loss_fn(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)

class REBertModel(nn.Module):
    def __init__(self, num_label):
        super().__init__()
        self.num_label = num_label
        self.bert = transformers.BertModel.from_pretrained(config.BASE_MODEL_PATH, return_dict=False)
        self.bert_drop_1 = nn.Dropout(0.3)

        # single layer
        self.out_label = nn.Linear(768, self.num_label)

        # multiple layer
        self.multi_layer = nn.Sequential(
            nn.Linear(768, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(32, self.num_label),
        )
        # lstm version
        self.lstm = nn.LSTM(input_size=768, hidden_size=768, num_layers=2, dropout=0.3, bidirectional=True, batch_first=True)
        self.bidi_dim = 768 * 2
        self.lstm_lin_layer = nn.Linear(self.bidi_dim, self.num_label)

    def forward(self, ids, mask, token_type_ids, targets):
        lstm_out, out = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)

        dropout_label = self.bert_drop_1(out)
        # single layer
        label = self.out_label(dropout_label)

        # multi layer
        # label = self.multi_layer(dropout_label)

        # lstm version
        # hidden, (last_hidden, last_cell) = self.lstm(lstm_out)
        # label = self.lstm_lin_layer(hidden[0])
        # if label.shape[0] != targets.shape[0]:
        #     label = label[0: targets.shape[0]]
        loss = loss_fn(label, targets)

        return label, loss
