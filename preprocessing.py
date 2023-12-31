import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import PorterStemmer

# ======================================================================================================================
# Note: Preprocessing implementation was taken from homework assignment 1
# ======================================================================================================================


def remove_stop_words_and_stemming(tokens):
    stop_words = set(nltk.corpus.stopwords.words("english"))
    stemmer = PorterStemmer()
    processed_and_stemmed_words = []

    for token in tokens:
        if token not in stop_words:
            processed_and_stemmed_words.append(stemmer.stem(token))
    return processed_and_stemmed_words


def remove_punctuation(text):
    str_without_punctuation = ""
    for index in range(len(text)):
        if not text[index].isalpha():
            str_without_punctuation = str_without_punctuation + ' '
        else:
            str_without_punctuation = str_without_punctuation + text[index]
    return str_without_punctuation

def remove_usernames(text):
    text_without_username = []

    for word in text.split(" "):
        if word.startswith('@') and len(word) > 1:
            continue
        else:
            text_without_username.append(word)

    return " ".join(text_without_username)

def remove_urls(text):
    text_without_username = []

    for word in text.split(" "):
        if word.startswith('http') and len(word) > 1:
            continue
        else:
            text_without_username.append(word)

    return " ".join(text_without_username)


def preprocess_data(
    unprocessed_data: pd.DataFrame
):
    # Step 1: lowercase letters
    unprocessed_data['preprocessed_text'] = unprocessed_data['text'].str.lower()

    # Step 2: Remove usernames:
    unprocessed_data['preprocessed_text'] = unprocessed_data['preprocessed_text'].apply(lambda row: remove_usernames(row))

    # Step 3: Remove urls:
    unprocessed_data['preprocessed_text'] = unprocessed_data['preprocessed_text'].apply(lambda row: remove_urls(row))

    # Step 4: Remove Numbers
    unprocessed_data['preprocessed_text'] = unprocessed_data['preprocessed_text'].str.replace('\d', '', regex=True)

    # Step 5: Remove punctuations:
    unprocessed_data['preprocessed_text'] = unprocessed_data['preprocessed_text'].apply(lambda row: remove_punctuation(row))

    # Step 6: tokenizing
    unprocessed_data['preprocessed_text'] = unprocessed_data['preprocessed_text'].apply(lambda row: nltk.word_tokenize(row))

    # Step 7: Remove stop words and stemming
    unprocessed_data['preprocessed_text'] = unprocessed_data['preprocessed_text'].apply(lambda tokens: remove_stop_words_and_stemming(tokens))

    return unprocessed_data