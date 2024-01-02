import sys
import io_util
import sentiment
import preprocessing
import pandas as pd

negative_preprocessed_data = None
neutral_preprocessed_data = None
positive_preprocessed_data = None


def print_help():
    print("possible actions".center(80, '='))
    print("[1] perform sentiment classification and preprocessing on dataset")
    print("[2] load preprocessed and classified data from csv-files")
    print("[3] send query")
    print("[4] quit program")
    print("".center(80, '='))


def perform_sentiment_classification():
    global negative_preprocessed_data, neutral_preprocessed_data, positive_preprocessed_data

    # TODO: replace True with False, this is only for testing purposes
    unprocessed_data = io_util.read_csv('data/training.1600000.processed.noemoticon.csv', ['id', 'text'], True)
    negative_unprocessed_data, neutral_unprocessed_data, positive_unprocessed_data \
        = sentiment.create_sentiment_dataframes(unprocessed_data)

    negative_preprocessed_data = preprocessing.preprocess_data(negative_unprocessed_data)
    neutral_preprocessed_data = preprocessing.preprocess_data(neutral_unprocessed_data)
    positive_preprocessed_data = preprocessing.preprocess_data(positive_unprocessed_data)

    io_util.write_csv('data/neutral_dataset.csv', negative_preprocessed_data)
    io_util.write_csv('data/negative_dataset.csv', neutral_preprocessed_data)
    io_util.write_csv('data/positive_dataset.csv', positive_preprocessed_data)


# After the classification of the large dataset has been performed once we can simply load the sentiment classification
# from the csv files that were created after the huge dataset has been classified and preprocessed so that the
# classification and preprocessing which take a lot of time do not have to be performed every time
def load_sentiment_classification():
    global negative_preprocessed_data, neutral_preprocessed_data, positive_preprocessed_data
    try:
        negative_preprocessed_data = io_util.read_csv('data/negative_dataset.csv', [], False)
        neutral_preprocessed_data = io_util.read_csv('data/neutral_dataset.csv', [], False)
        positive_preprocessed_data = io_util.read_csv('data/positive_dataset.csv', [], False)
    except FileNotFoundError:
        print("Files to load sentiment classification do not exist!")

def send_query():
    global negative_preprocessed_data, neutral_preprocessed_data, positive_preprocessed_data

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

    print(preprocessed_query)

    if ir_method == 'bm25':
        # TODO: add bm25 IR here
        pass
    elif ir_method == 'tfidf':
        # TODO add tfidf IR here
        pass



print("Program started".center(80, "="))
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
        break
    else:
        print("Invalid operation!")

print("Program finished".center(80, "="))