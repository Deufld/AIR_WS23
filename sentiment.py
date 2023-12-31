import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax

# ======================================================================================================================
# Source: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
# ======================================================================================================================

# run calculation on GPU to speed up calculation
device = "cuda:0" if torch.cuda.is_available() else "cpu"

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model = model.to(device)


def create_sentiment_dataframes(
    unprocessed_data: pd.DataFrame
):
    negative_unprocessed_data = pd.DataFrame(columns=['text'])
    neutral_unprocessed_data = pd.DataFrame(columns=['text'])
    positive_unprocessed_data = pd.DataFrame(columns=['text'])

    print("classification process started".center(80, "="))

    for index, row in unprocessed_data.iterrows():

        if index % int(len(unprocessed_data) / 10) == 0 and index != 0:
            print(str(index / int(len(unprocessed_data) / 10) * 10) + "% of classification has been completed")

        text = row['text']

        sentiment = get_sentiment_of_text(text)
        if sentiment == config.id2label[0]:
            negative_unprocessed_data.loc[len(negative_unprocessed_data)] = row
        elif sentiment == config.id2label[1]:
            neutral_unprocessed_data.loc[len(neutral_unprocessed_data)] = row
        else:
            positive_unprocessed_data.loc[len(positive_unprocessed_data)] = row

    print("classification has been completed".center(80, "="))

    return negative_unprocessed_data, neutral_unprocessed_data, positive_unprocessed_data


def get_sentiment_of_text(
    text: str
):
    encoded_text = tokenizer(text, return_tensors='pt').to(device)
    result = model(**encoded_text)
    scores = result[0][0].cpu().detach().numpy()
    scores = list(softmax(scores))
    index_of_most_likely_sentiment = scores.index(max(scores))

    return config.id2label[index_of_most_likely_sentiment]
