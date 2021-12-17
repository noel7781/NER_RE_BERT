import config
import torch
import torch.nn as nn
import transformers

def loss_fn(output, target, mask, num_labels):
    loss_function = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
            active_loss,
            target.view(-1),
            torch.tensor(loss_function.ignore_index).type_as(target)
    )
    # print("active logit shape: ", active_logits.shape)
    # print("active label shape: ", active_labels.shape)
    loss = loss_function(active_logits, active_labels)
    return loss

class NERBertModel(nn.Module):
    def __init__(self, num_tag):
        super().__init__()
        self.num_tag = num_tag
        self.bert = transformers.BertModel.from_pretrained(config.BASE_MODEL_PATH, return_dict=False)
        self.bert_drop_1 = nn.Dropout(0.3)

        # single layer
        self.out_tag = nn.Linear(768, self.num_tag)

        # multiple layer
        self.multi_layer = nn.Sequential(
            nn.Linear(768, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(32, self.num_tag),
        )

        # lstm version
        self.lstm = nn.LSTM(input_size=768, hidden_size=768, num_layers=2, dropout=0.3, bidirectional=True, batch_first=True)
        self.bidi_dim = 768 * 2
        self.lstm_lin_layer = nn.Linear(self.bidi_dim, self.num_tag)

    def forward(self, ids, mask, token_type_ids, target_tag, token_starts):
        out, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)

        dropout_tag = self.bert_drop_1(out)

        # single version
        tag = self.out_tag(dropout_tag)

        # multi version
        # tag = self.multi_layer(dropout_tag)


        ## lstm version
        # hidden, (last_hidden, last_cell) = self.lstm(out)
        # lstm_tag = self.lstm_lin_layer(hidden)
        # tag = lstm_tag

        loss_tag = loss_fn(tag, target_tag, mask, self.num_tag)
        loss = loss_tag

        return tag, loss
