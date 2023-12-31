import sys
import io_util
import sentiment
import preprocessing

negative_preprocessed_data = None
neutral_preprocessed_data = None
positive_preprocessed_data = None


def print_help():
    print("possible actions".center(80, '='))
    print("[1] perform sentiment classification and preprocessing on dataset")
    print("[2] load preprocessed and classified data from csv-files")
    print("[3] quit program")
    print("".center(80, '='))


def perform_sentiment_classification():
    global negative_preprocessed_data, neutral_preprocessed_data, positive_preprocessed_data

    # TODO: replace True with False, this is only for testing purposes
    unprocessed_data = io_util.read_csv('data/training.1600000.processed.noemoticon.csv', ['text'], True)
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

        print(negative_preprocessed_data)
        print(neutral_preprocessed_data)
        print(str(len(negative_preprocessed_data) + len(neutral_preprocessed_data) + len(positive_preprocessed_data)))
    except FileNotFoundError:
        print("Files to load sentiment classification do not exist!")


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
        break
    else:
        print("Invalid operation!")

print("Program finished".center(80, "="))