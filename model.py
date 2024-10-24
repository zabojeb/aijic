import torch
from transformers import XLMRobertaTokenizer, AutoModelForSequenceClassification
import pandas as pd

model_path = "zabojeb/rubert-classifier"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path, config=f"{model_path}/config.json"
)
model.eval()


def review(text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    with torch.no_grad():
        logits = model(**inputs)

    probabilities = torch.sigmoid(logits.logits).cpu().numpy()
    pred_labels = (probabilities > 0.5).astype(int)
    return pred_labels[0]


def reviews(df):
    for i in range(df.shape[0]):
        inputs = tokenizer(
            df["Reviews"][i],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        with torch.no_grad():
            logits = model(**inputs)

        probabilities = torch.sigmoid(logits.logits).cpu().numpy()
        pred_labels = (probabilities > 0.5).astype(int)
        pred_labels = [int(i) for i in pred_labels[0]]
        for j in range(5):
            aspects = [
                "практика",
                "теория",
                "преподаватель",
                "технологии",
                "актуальность",
            ]
            df.loc[i, aspects[j]] = pred_labels[j]
    return df


def ton_review(text):
    """
    Функция для анализа тональности в тексте.
    Возвращает массив из 5 чисел.
    """

    return [0, 0, 0, 0, 0]


def string_analyse(text):
    result = list(review(text))
    aspects = ["практика", "теория", "преподаватель", "технологии", "актуальность"]
    response = dict((aspects[i], int(result[i])) for i in range(5))
    return response


def ton_analyse(text):
    result = list(ton_review(text))
    aspects = ["практика", "теория", "преподаватель", "технологии", "актуальность"]
    response = dict((aspects[i], int(result[i])) for i in range(5))


def tests_analyse(df):
    return reviews(df)

model_checkpoint = 'cointegrated/rubert-base-cased-nli-threeway'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
if torch.cuda.is_available():
    model.cuda()

def predict_zero_shot(text, label_texts, model, tokenizer, label='entailment', normalize=True):
    label_texts
    tokens = tokenizer([text] * len(label_texts), label_texts, truncation=True, return_tensors='pt', padding=True)
    with torch.inference_mode():
        result = torch.softmax(model(**tokens.to(model.device)).logits, -1)
    proba = result[:, model.config.label2id[label]].cpu().numpy()
    if normalize:
        proba /= sum(proba)
    return proba
aspects1 = ['практика','теория','преподаватель','технологии','актуальность']
