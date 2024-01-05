from tabulate import tabulate
import io_util
import sentiment
import preprocessing
import pandas as pd

from IR.bm25 import BM25
from IR.tfidf import TFIDF
from IR.bert import BERT
from evaluation import Evaluation

# Datasets
negative_preprocessed_data = None
neutral_preprocessed_data = None
positive_preprocessed_data = None
negative_positive_preprocessed_data = None
negative_neutral_preprocessed_data = None
positive_neutral_preprocessed_data = None
negative_positive_neutral_preprocessed_data = None

# TF-IDF Instances
negative_tfidf = None
neutral_tfidf = None
positive_tfidf = None
negative_positive_tfidf = None
negative_neutral_tfidf = None
positive_neutral_tfidf = None
negative_positive_neutral_tfidf = None

# BM25 Instances
negative_bm25 = None
neutral_bm25 = None
positive_bm25 = None
negative_positive_bm25 = None
negative_neutral_bm25 = None
positive_neutral_bm25 = None
negative_positive_neutral_bm25 = None


def print_help():
    print("possible actions".center(80, '='))
    print("[1] perform sentiment classification and preprocessing on dataset")
    print("[2] load preprocessed and classified data from csv-files")
    print("[3] send query")
    print("[4] do evaluation")
    print("[5] quit program")
    print("".center(80, '='))


def perform_sentiment_classification():
    global negative_preprocessed_data, neutral_preprocessed_data, positive_preprocessed_data
    global negative_positive_preprocessed_data, negative_neutral_preprocessed_data, positive_neutral_preprocessed_data
    global positive_neutral_preprocessed_data, negative_positive_neutral_preprocessed_data
    global negative_tfidf, neutral_tfidf, positive_tfidf
    global negative_bm25, neutral_bm25, positive_bm25
    global negative_positive_tfidf, negative_neutral_tfidf, positive_neutral_tfidf, negative_positive_neutral_tfidf
    global negative_positive_bm25, negative_neutral_bm25, positive_neutral_bm25, negative_positive_neutral_bm25

    # TODO: replace True with False, this is only for testing purposes
    unprocessed_data = io_util.read_csv('data/training.1600000.processed.noemoticon.csv', ['id', 'text'], True)
    negative_unprocessed_data, neutral_unprocessed_data, positive_unprocessed_data = sentiment.create_sentiment_dataframes(
        unprocessed_data)

    negative_preprocessed_data = preprocessing.preprocess_data(negative_unprocessed_data)
    neutral_preprocessed_data = preprocessing.preprocess_data(neutral_unprocessed_data)
    positive_preprocessed_data = preprocessing.preprocess_data(positive_unprocessed_data)

    negative_positive_preprocessed_data = pd.concat([negative_preprocessed_data, positive_preprocessed_data])
    negative_neutral_preprocessed_data = pd.concat([negative_preprocessed_data, neutral_preprocessed_data])
    positive_neutral_preprocessed_data = pd.concat([positive_preprocessed_data, neutral_preprocessed_data])
    negative_positive_neutral_preprocessed_data = pd.concat(
        [negative_preprocessed_data, positive_preprocessed_data, neutral_preprocessed_data])

    # TF-IDF
    negative_tfidf = TFIDF(["negative"], negative_preprocessed_data, False)
    neutral_tfidf = TFIDF(["neutral"], neutral_preprocessed_data, False)
    positive_tfidf = TFIDF(["positive"], positive_preprocessed_data, False)
    negative_positive_tfidf = TFIDF(["negative", "positive"], negative_positive_preprocessed_data, False)
    negative_neutral_tfidf = TFIDF(["negative", "neutral"], negative_neutral_preprocessed_data, False)
    positive_neutral_tfidf = TFIDF(["positive", "neutral"], positive_neutral_preprocessed_data, False)
    negative_positive_neutral_tfidf = TFIDF(["negative", "positive", "neutral"],
                                            negative_positive_neutral_preprocessed_data, False)

    # BM25
    negative_bm25 = BM25(["negative"], negative_preprocessed_data, False)
    neutral_bm25 = BM25(["neutral"], neutral_preprocessed_data, False)
    positive_bm25 = BM25(["positive"], positive_preprocessed_data, False)
    negative_positive_bm25 = BM25(["negative", "positive"], negative_positive_preprocessed_data, False)
    negative_neutral_bm25 = BM25(["negative", "neutral"], negative_neutral_preprocessed_data, False)
    positive_neutral_bm25 = BM25(["positive", "neutral"], positive_neutral_preprocessed_data, False)
    negative_positive_neutral_bm25 = BM25(["negative", "positive", "neutral"],
                                          negative_positive_neutral_preprocessed_data, False)

    io_util.write_csv('data/negative_dataset.csv', negative_preprocessed_data)
    io_util.write_csv('data/neutral_dataset.csv', neutral_preprocessed_data)
    io_util.write_csv('data/positive_dataset.csv', positive_preprocessed_data)


# After the classification of the large dataset has been performed once we can simply load the sentiment classification
# from the csv files that were created after the huge dataset has been classified and preprocessed so that the
# classification and preprocessing which take a lot of time do not have to be performed every time
def load_sentiment_classification():
    global negative_preprocessed_data, neutral_preprocessed_data, positive_preprocessed_data
    global negative_positive_preprocessed_data, negative_neutral_preprocessed_data, positive_neutral_preprocessed_data
    global positive_neutral_preprocessed_data, negative_positive_neutral_preprocessed_data
    global negative_tfidf, neutral_tfidf, positive_tfidf
    global negative_bm25, neutral_bm25, positive_bm25
    global negative_positive_tfidf, negative_neutral_tfidf, positive_neutral_tfidf, negative_positive_neutral_tfidf
    global negative_positive_bm25, negative_neutral_bm25, positive_neutral_bm25, negative_positive_neutral_bm25

    try:
        negative_preprocessed_data = io_util.read_csv('data/negative_dataset.csv', [], False)
        neutral_preprocessed_data = io_util.read_csv('data/neutral_dataset.csv', [], False)
        positive_preprocessed_data = io_util.read_csv('data/positive_dataset.csv', [], False)

        negative_positive_preprocessed_data = pd.concat([negative_preprocessed_data, positive_preprocessed_data])
        negative_neutral_preprocessed_data = pd.concat([negative_preprocessed_data, neutral_preprocessed_data])
        positive_neutral_preprocessed_data = pd.concat([positive_preprocessed_data, neutral_preprocessed_data])
        negative_positive_neutral_preprocessed_data = pd.concat(
            [negative_preprocessed_data, positive_preprocessed_data, neutral_preprocessed_data])

        # TF-IDF
        negative_tfidf = TFIDF(["negative"], negative_preprocessed_data, True)
        neutral_tfidf = TFIDF(["neutral"], neutral_preprocessed_data, True)
        positive_tfidf = TFIDF(["positive"], positive_preprocessed_data, True)
        negative_positive_tfidf = TFIDF(["negative", "positive"], negative_positive_preprocessed_data, True)
        negative_neutral_tfidf = TFIDF(["negative", "neutral"], negative_neutral_preprocessed_data, True)
        positive_neutral_tfidf = TFIDF(["positive", "neutral"], positive_neutral_preprocessed_data, True)
        negative_positive_neutral_tfidf = TFIDF(["negative", "positive", "neutral"],
                                                negative_positive_neutral_preprocessed_data, True)

        # BM25
        negative_bm25 = BM25(["negative"], negative_preprocessed_data, True)
        neutral_bm25 = BM25(["neutral"], neutral_preprocessed_data, True)
        positive_bm25 = BM25(["positive"], positive_preprocessed_data, True)
        negative_positive_bm25 = BM25(["negative", "positive"], negative_positive_preprocessed_data, True)
        negative_neutral_bm25 = BM25(["negative", "neutral"], negative_neutral_preprocessed_data, True)
        positive_neutral_bm25 = BM25(["positive", "neutral"], positive_neutral_preprocessed_data, True)
        negative_positive_neutral_bm25 = BM25(["negative", "positive", "neutral"],
                                              negative_positive_neutral_preprocessed_data, True)

    except FileNotFoundError:
        print("Files to load sentiment classification do not exist!")


def send_query():
    global negative_preprocessed_data, neutral_preprocessed_data, positive_preprocessed_data
    global negative_positive_preprocessed_data, negative_neutral_preprocessed_data, positive_neutral_preprocessed_data
    global positive_neutral_preprocessed_data, negative_positive_neutral_preprocessed_data
    global negative_tfidf, neutral_tfidf, positive_tfidf
    global negative_positive_tfidf, negative_neutral_tfidf, positive_neutral_tfidf, negative_positive_neutral_tfidf

    if negative_preprocessed_data is None or neutral_preprocessed_data is None or positive_preprocessed_data is None:
        print("Cannot perform query, dataset has not yet been loaded / classified!")
        return

    ir_method = input("Enter IR-Method of choice [bm25;tfidf]: ")
    if ir_method not in ['bm25', 'tfidf']:
        print("Invalid/not supported IR-Method!")
        return

    sentiment_of_choice = input("Enter sentiment(s) of choice (separated by semicolon) [negative, neutral, positive]: ")
    sentiments = sentiment_of_choice.split(";")

    for s in sentiments:
        if s not in ['positive', 'neutral', 'negative']:
            print("Invalid / not supported sentiment was entered!")
            return

    query_str = input("Enter query: ")
    query_as_df = pd.DataFrame(data=[query_str], columns=['text'])
    preprocessed_query = preprocessing.preprocess_data(query_as_df)

    try:
        n = int(input("Enter the number of documents you want to retrieve: "))
        if n <= 0:
            raise ValueError
    except ValueError:
        print("Enter a valid integer for the number of documents > 0!")
        return

    print("Searching documents for query: " + query_str)

    retrieved_documents = None
    if ir_method == 'bm25':
        if 'negative' in sentiments and 'positive' in sentiments and 'neutral' in sentiments:
            retrieved_documents = negative_positive_neutral_bm25.retrieve_n_most_relevant_documents(preprocessed_query,
                                                                                                    n)
        elif 'negative' in sentiments and 'positive' in sentiments:
            retrieved_documents = negative_positive_bm25.retrieve_n_most_relevant_documents(preprocessed_query, n)
        elif 'positive' in sentiments and 'neutral' in sentiments:
            retrieved_documents = positive_neutral_bm25.retrieve_n_most_relevant_documents(preprocessed_query, n)
        elif 'negative' in sentiments and 'neutral' in sentiments:
            retrieved_documents = negative_neutral_bm25.retrieve_n_most_relevant_documents(preprocessed_query, n)
        elif 'positive' in sentiments:
            retrieved_documents = positive_bm25.retrieve_n_most_relevant_documents(preprocessed_query, n)
        elif 'negative' in sentiments:
            retrieved_documents = negative_bm25.retrieve_n_most_relevant_documents(preprocessed_query, n)
        elif 'neutral' in sentiments:
            retrieved_documents = neutral_bm25.retrieve_n_most_relevant_documents(preprocessed_query, n)
        else:
            print("This is really odd!")

    elif ir_method == 'tfidf':
        if 'negative' in sentiments and 'positive' in sentiments and 'neutral' in sentiments:
            retrieved_documents = negative_positive_neutral_tfidf.retrieve_n_most_relevant_documents(preprocessed_query,
                                                                                                     n)
        elif 'negative' in sentiments and 'positive' in sentiments:
            retrieved_documents = negative_positive_tfidf.retrieve_n_most_relevant_documents(preprocessed_query, n)
        elif 'positive' in sentiments and 'neutral' in sentiments:
            retrieved_documents = positive_neutral_tfidf.retrieve_n_most_relevant_documents(preprocessed_query, n)
        elif 'negative' in sentiments and 'neutral' in sentiments:
            retrieved_documents = negative_neutral_tfidf.retrieve_n_most_relevant_documents(preprocessed_query, n)
        elif 'positive' in sentiments:
            retrieved_documents = positive_tfidf.retrieve_n_most_relevant_documents(preprocessed_query, n)
        elif 'negative' in sentiments:
            retrieved_documents = negative_tfidf.retrieve_n_most_relevant_documents(preprocessed_query, n)
        elif 'neutral' in sentiments:
            retrieved_documents = neutral_tfidf.retrieve_n_most_relevant_documents(preprocessed_query, n)
        else:
            print("This is really odd!")

    print(
        tabulate(retrieved_documents, headers='keys', tablefmt='grid', showindex=False, maxcolwidths=[10, 34, 10, 10]))

    bert_option = input("Enter [yes;no] if you want the results to be re-ranked with BERT: ")
    if bert_option not in ['yes', 'no']:
        print("Invalid/not supported answer for BERT!")
        return

    if bert_option == 'yes':
        output_bert_df = bert.rerank_with_bert(retrieved_documents, query_str, n)

        print(f"Query: {query_str}")
        print(f"\nTop {n} most similar sentences in corpus:")
        print(tabulate(output_bert_df, headers='keys', tablefmt='grid', showindex=False, maxcolwidths=[10, 50, 10]))
    elif bert_option == 'no':
        return


def do_evaluation():
    if negative_bm25 is None:
        print("Do [1] or [2] first!")
        return

    evaluation = Evaluation(negative_bm25, neutral_bm25, positive_bm25, negative_tfidf, neutral_tfidf, positive_tfidf)
    evaluation.perform_evaluation()


print("Program started".center(80, "="))

# initialize BERT
bert = BERT()
while True:
    print_help()
    try:
        operation = int(input("Enter action to perform: "))
    except ValueError:
        print("Invalid operation!")

    if operation == 1:
        perform_sentiment_classification()
    elif operation == 2:
        load_sentiment_classification()
    elif operation == 3:
        send_query()
    elif operation == 4:
        do_evaluation()
    elif operation == 5:
        break
    else:
        print("Invalid operation!")

print("Program finished".center(80, "="))
