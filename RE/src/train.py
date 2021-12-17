import pandas as pd
import numpy as np

from tqdm import tqdm

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

from model import REBertModel

import config
import dataset

import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup

from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_one_step(model, data, optimizer, device):
    optimizer.zero_grad()
    for k, v in data.items():
        data[k] = v.to(device)
    _, loss = model(**data)
    loss.backward()
    optimizer.step()
    return loss

def train_one_epoch(data_loader, model, optimizer, device, scheduler):
    model.train()
    total_loss = 0.0
    for data in tqdm(data_loader, total=len(data_loader)):
        loss = train_one_step(model, data, optimizer, device)
        scheduler.step()
        total_loss += loss
    return total_loss / len(data_loader)

def eval_one_step(model, data, device):
    for k, v in data.items():
        data[k] = v.to(device)
    _, loss = model(**data)
    return loss


def eval_one_epoch(data_loader, model, device):
    model.eval()
    total_loss = 0.0
    for data in tqdm(data_loader, total=len(data_loader)):
        loss = eval_one_step(model, data, device)
        total_loss += loss
    return total_loss / len(data_loader)

def preprocess_data(data_path):
    df = pd.read_csv(data_path, skiprows=[0],delimiter='\t', names=["index", "sentence", "label"], header=None)

    enc_label = preprocessing.LabelEncoder()
    
    df.loc[:, "label"] = enc_label.fit_transform(df["label"])

    sentences = df["sentence"]
    label = df["label"]
    sentences = np.array(sentences)
    label = np.array(label)
    return sentences, label, enc_label

def get_metrics_score(y_test, pred, epoch):
    # chemprot -> micro // DDI -> macro
    y_test = np.array(y_test)
    pred = np.array(pred)
    print("y_test:", y_test)
    print("pred:", pred)
    average = 'macro'
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average=average)
    recall = recall_score(y_test, pred, average=average)
    f1 = f1_score(y_test, pred, average=average)
    writer.add_scalar(f"f1 score_{average}", f1, epoch)
    y_test = pd.DataFrame(y_test)
    pred = pd.DataFrame(pred)
    y_test.to_csv("output/gold_file.csv")
    pred.to_csv("output/predict.csv")
    print("test label count:", len(np.unique(np.array(pred_tags))))

    print("Bert-base-uncased Linear layer Metrics")
    print(f"accuracy:{accuracy}\n{average} precision:{precision}\n{average} recall:{recall}\n{average} f1:{f1} epoch:{epoch}")

if __name__ == "__main__":
    # sentences, label, enc_label = preprocess_data(config.TRAINING_FILE)
    train_sentences, train_label, enc_label = preprocess_data(config.TRAINING_FILE)
    valid_sentences, valid_label, val_enc_label = preprocess_data(config.VALIDATING_FILE)

    writer = SummaryWriter(f"runs/{config.BASE_MODEL_PATH}_MAXLEN_{config.MAX_LEN}_EPOCH_{config.EPOCHS}_ddi")
    
    train_sentences, valid_sentences, train_label, valid_label = model_selection.train_test_split(sentences, label, random_state=18, test_size=0.01)
    
    num_label = len(list(enc_label.classes_))

    train_dataset = dataset.REDataset(texts=train_sentences, labels=train_label)
    valid_dataset = dataset.REDataset(texts=valid_sentences, labels=valid_label)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, num_workers=2, shuffle=True)#, drop_last=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, num_workers=1)
    
    test_sentences, test_label, test_enc_label = preprocess_data(config.TESTING_FILE)
    test_num_label = len(list(test_enc_label.classes_))

    test_dataset = dataset.REDataset(texts=test_sentences, labels=test_label)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE, num_workers=1)#, drop_last=True)
  
    device = torch.device(config.DEVICE)
    model = REBertModel(num_label=num_label)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "layernorm.bias", "layernorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                param for n, param in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                param for n, param in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_sentences) / config.BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_loss = np.inf
    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(train_dataloader, model, optimizer, device, scheduler)
        valid_loss = eval_one_epoch(valid_dataloader, model, device)
        print(f"train loss = {train_loss} valid loss= {valid_loss}")
        writer.add_scalar("Training Loss", train_loss, epoch+1)
        writer.add_scalar("Validation Loss", valid_loss, epoch+1)
        if valid_loss < best_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = valid_loss

        model.eval()
        pred_tags = []
        for data in tqdm(test_dataloader, total=len(test_dataloader)):
            for k, v in data.items():
                data[k] = v.to(device)
            label, _ = model(**data)
            pred_tags.extend(torch.argmax(label, axis=1).cpu().detach().numpy())



    del(model)


    device = torch.device(config.DEVICE)
    model = REBertModel(num_label=test_num_label)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)
    pred_tags = []

    model.eval()
    for data in tqdm(test_dataloader, total=len(test_dataloader)):
        for k, v in data.items():
            data[k] = v.to(device)
        label, _ = model(**data)
        pred_tags.extend(test_enc_label.inverse_transform(torch.argmax(label, axis=1).cpu().detach().numpy()))

    test_label = test_enc_label.inverse_transform(test_label)
    get_metrics_score(test_label, pred_tags)
