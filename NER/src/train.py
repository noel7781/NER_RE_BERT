import pandas as pd
import numpy as np

from tqdm import tqdm

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

from model import NERBertModel

import config
import dataset

import torch
from transformers import AdamW, get_linear_schedule_with_warmup

from tensorboardX import SummaryWriter

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
    df = pd.read_csv(data_path, delimiter='\t', names=["text", "Paragraph", "pos", "tag"], header=None)
    start = 1
    for index, row in df.iterrows():
        if row["Paragraph"].isdigit():
            df.loc[index, "Paragraph"] = start
            start += 1
    df["Paragraph"] = df["Paragraph"].replace('-', np.NaN)
    df.loc[:, "Paragraph"] = df["Paragraph"].fillna(method="ffill")
    enc_pos = preprocessing.LabelEncoder()
    enc_tag = preprocessing.LabelEncoder()

    df["text"] = df["text"].dropna()
    df.loc[:, "pos"] = enc_pos.fit_transform(df["pos"])
    df.loc[:, "tag"] = enc_tag.fit_transform(df["tag"])

    df["text"] = df["text"].dropna()

    sentences = df.groupby("Paragraph")["text"].apply(list).values
    pos = df.groupby("Paragraph")["pos"].apply(list).values
    tag = df.groupby("Paragraph")["tag"].apply(list).values
    return df, sentences, pos, tag, enc_pos, enc_tag

if __name__ == "__main__":
    df, sentences, pos, tag, enc_pos, enc_tag = preprocess_data(config.TRAINING_FILE)
    test_df, test_sentences, test_pos, test_tag, test_enc_pos, test_enc_tag = preprocess_data(config.TESTING_FILE)

    writer = SummaryWriter("bert")
    
    num_tag = len(list(enc_tag.classes_))
    test_num_tag = len(list(test_enc_tag.classes_))

    train_sentences, valid_sentences, train_pos, valid_pos, train_tag, valid_tag = model_selection.train_test_split(sentences, pos, tag, random_state=77, test_size=0.1)
    train_tag = tag

    train_dataset = dataset.NERDataset(texts=train_sentences, tags=train_tag)
    valid_dataset = dataset.NERDataset(texts=valid_sentences, tags=valid_tag)
    test_dataset = dataset.NERDataset(texts=test_sentences, tags=test_tag)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, num_workers=4)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, num_workers=1)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE, num_workers=1)

    device = torch.device(config.DEVICE)
    model = NERBertModel(num_tag=num_tag)
    model.to(device)


    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
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
        torch.save(model.state_dict(), config.MODEL_PATH)
        if valid_loss < best_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = valid_loss

    del(model)

    print("Delete model and reload best state")
    device = torch.device(config.DEVICE)
    model = NERBertModel(num_tag=test_num_tag)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)

    pred_tags = []
    for sentence in test_sentences:
        tokenized_sentence = config.TOKENIZER.encode(str(sentence), is_split_into_words=True)
        simple_dataset = dataset.NERDataset(
                texts=[sentence],
                tags=[[0] * len(sentence)],
                )
        with torch.no_grad():
            data = simple_dataset[0]
            for k, v in data.items():
                data[k] = v.to(device).unsqueeze(0)
            tag,  _ = model(**data)
            bio_tags = enc_tag.inverse_transform(tag.argmax(2).cpu().numpy().reshape(-1))
            start_point = data["token_starts"].cpu().numpy()[0]
            start_point = start_point[start_point > 0]
            pred_tag = [bio_tags[i] if i < config.MAX_LEN else 'O' for i in start_point]
            ov_length = len(sentence) - len(start_point)
            if ov_length > 0:
                print("OV length:",ov_length)
            pred_tag += ('O' * ov_length)
        pred_tags.extend(pred_tag)

    df_predict = pd.read_csv(config.TESTING_FILE, delimiter='\t', names=["text", "Paragraph", "pos", "tag"], header=None)
    test_df = df_predict.copy()
    df_predict.loc[:, "tag"] = pred_tags
    test_df.to_csv("output/gold_file.tsv", index=False, sep='\t', header=False)
    df_predict.to_csv("output/pred_output.tsv", index=False, sep='\t', header=False)

    test_df.to_json("output/gold_file.json", orient = 'columns')
    df_predict.to_json("output/pred_output.json", orient = 'columns')
