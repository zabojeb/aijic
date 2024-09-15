import torch
from transformers import XLMRobertaTokenizer, AutoModelForSequenceClassification
import pandas as pd


def review(text):
    model_path = "model\\fine_tuned_rubert"
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, config=f'{model_path}\\config.json')
    model.eval()

    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs)

    probabilities = torch.sigmoid(logits.logits).cpu().numpy()
    pred_labels = (probabilities > 0.5).astype(int)
    return pred_labels[0]


def reviews(df):
    model_path = "model\\fine_tuned_rubert"
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, config=f'{model_path}\\config.json')
    model.eval()
    for i in range(df.shape[0]):
        inputs = tokenizer(df['Reviews'][i], return_tensors="pt",
                           truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs)

        probabilities = torch.sigmoid(logits.logits).cpu().numpy()
        pred_labels = (probabilities > 0.5).astype(int)
        pred_labels = [int(i) for i in pred_labels[0]]
        for j in range(5):
            aspects = ["практика", "теория", "преподаватель",
                       "технологии", "актуальность"]
            df.loc[i, aspects[j]] = pred_labels[j]
    return df


def string_analyse(text):
    result = list(review(text))
    aspects = ["практика", "теория", "преподаватель",
               "технологии", "актуальность"]
    response = dict((aspects[i], int(result[i])) for i in range(5))
    return response


def tests_analyse(df):
    return reviews(df)
