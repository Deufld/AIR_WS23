# Twitter Sentiment-Based Information Retrieval & Analysis

### Dataset
We used the following dataset https://www.kaggle.com/datasets/kazanova/sentiment140/ which contained 1.6 million tweets.

### Classification
Within our application we than used the following pre-trained model in order to classify each tweet into one of the three categories {Negative, Neutral, Positive}: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

### IR-Methods
We used BM25 and TF-IDF as our IR-methods and for TF-IDF we decided on using the advanced framework Gensim (https://radimrehurek.com/gensim/intro.html) to help us with RAM efficiency.

### Re-ranking via BERT
The documents that are retrieved via TF-IDF or BM25 can be re-ranked via BERT.

### Program Description
We created a CLI application which allows the user to perform the following actions:
* **[1] perform sentiment classification and preprocessing on dataset**
  * This action has to be performed when the user starts the program for the first time. Selecting this action performs the sentiment-based classification on the dataset, pre-processes the tweets and creates the necessary infrastructure for TF-IDF / BM25. The results of the classification as well as the infrastructure for TF-IDF / BM25 are stored in appropriate files such that the user only has to perform this step once.
* **[2] load preprocessed and classified data from csv-files**
  * As already mentioned, the results of the classification as well as the infrastructure for TF-IDF and BM25 are stored in appropriate files if the user has previously performed action [1]. If those files exists the user can use this action in order to load the classification as well as the infrastructure for TF-IDF and BM25 from those files rather than performing the very time-consuming classification again.
* **[3] send query**
  * This operation allows the user to send a query and retrieve documents. The user can choose between TF-IDF and BM25 as the IR-method, select one or more sentiments (negative, neutral, positive) and whether or not the results should be re-ranked via BERT.
* **[4] do evaluation**
  * This action firstly calculates / generates the ground-truth on basis of the queries within negative_queries.csv, neutral_queries.csv and positive_queries.csv files. After the ground-truth has been calculated the precision, recall and F1-score are calculated for:
    * **TFIDF**
    * **BM25**
    * **TFIDF + re-ranking via BERT**
    * **BM25 + re-ranking via BERT**
  * Performing this evaluation is a great indicator for the effectiveness of re-ranking via BERT.
* **[5] quit program**
  * Quits the program
### Setup
In order to use the CLI application you have to download the above-mentioned dataset (https://www.kaggle.com/datasets/kazanova/sentiment140/) and install all the necessary libraries that are not yet installed on your local system. After those steps have been completed, the program can than be started by running the **main.py** file.
