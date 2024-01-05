from tabulate import tabulate
import io_util
import preprocessing
import pandas as pd
import os.path
from IR.bert import BERT
import csv


class Evaluation:
    def __init__(self, negative_bm25, neutral_bm25, positive_bm25, negative_tfidf, neutral_tfidf, positive_tfidf):
        self.negative_queries = io_util.read_csv('data/negative_queries.csv', [], False)
        self.neutral_queries = io_util.read_csv('data/neutral_queries.csv', [], False)
        self.positive_queries = io_util.read_csv('data/positive_queries.csv', [], False)

        self.negative_bm25 = negative_bm25
        self.neutral_bm25 = neutral_bm25
        self.positive_bm25 = positive_bm25

        self.negative_tfidf = negative_tfidf
        self.neutral_tfidf = neutral_tfidf
        self.positive_tfidf = positive_tfidf

        self.ground_truth_dict_bm25 = dict()
        self.ground_truth_dict_tfidf = dict()

        self.bert = BERT()

    def generate_ground_truth(self, queries, object_for_calculation, dict_to_fill):
        for index, row in queries.iterrows():
            qid = row['qid']
            query_str = row['text']
            query_as_df = pd.DataFrame(data=[query_str], columns=['text'])
            preprocessed_query = preprocessing.preprocess_data(query_as_df)
            retrieved_documents = object_for_calculation.retrieve_n_most_relevant_documents(preprocessed_query, 10)
            output_bert_df = self.bert.rerank_with_bert(retrieved_documents, query_str, 5)
            dict_to_fill[qid] = output_bert_df['id'].values.tolist()

    def read_ground_truth_csv(self, filename):
        return_dict = dict()
        with open(filename) as f:
            reader = csv.reader(f, delimiter=',')
            # skip header
            next(reader)
            for row in reader:
                qid = int(row[0])
                tmp_list = []
                for field in row[1:]:
                    only_numbers = int(''.join(filter(str.isdigit, field)))
                    tmp_list.append(only_numbers)
                return_dict[qid] = tmp_list
        return return_dict

    def perform_evaluation(self):
        if os.path.exists('data/evaluation_results.csv'):
            with open('data/evaluation_results.csv') as f:
                reader = csv.reader(f)
                for row in reader:
                    print(", ".join(row))
            return
        else:
            if os.path.exists('data/ground_truth_bm25.csv') and os.path.exists('data/ground_truth_tfidf.csv'):
                self.ground_truth_dict_bm25 = self.read_ground_truth_csv('data/ground_truth_bm25.csv')
                self.ground_truth_dict_tfidf = self.read_ground_truth_csv('data/ground_truth_tfidf.csv')
            else:
                self.generate_ground_truth(self.negative_queries, self.negative_bm25, self.ground_truth_dict_bm25)
                self.generate_ground_truth(self.neutral_queries, self.neutral_bm25, self.ground_truth_dict_bm25)
                self.generate_ground_truth(self.positive_queries, self.positive_bm25, self.ground_truth_dict_bm25)

                self.generate_ground_truth(self.negative_queries, self.negative_tfidf, self.ground_truth_dict_tfidf)
                self.generate_ground_truth(self.neutral_queries, self.neutral_tfidf, self.ground_truth_dict_tfidf)
                self.generate_ground_truth(self.positive_queries, self.positive_tfidf, self.ground_truth_dict_tfidf)

                with open('data/ground_truth_bm25.csv', 'a') as csv_file:
                    csv_file.write('qid,docids\n')
                    [csv_file.write('{0},{1}\n'.format(key, value)) for key, value in self.ground_truth_dict_bm25.items()]

                with open('data/ground_truth_tfidf.csv', 'a') as csv_file:
                    csv_file.write('qid,docids\n')
                    [csv_file.write('{0},{1}\n'.format(key, value)) for key, value in self.ground_truth_dict_tfidf.items()]

        self.evaluate_bm25()
        self.evaluate_bm25_with_bert()
        self.evaluate_tfidf()
        self.evaluate_tfidf_with_bert()

    def write_to_evaluation_csv(self, function_name, precision, recall, f1):
        with open('data/evaluation_results.csv', 'a') as f:
            # Define the data to be written
            data = [f"{function_name}: Precision: {precision}, Recall: {recall}, F1: {f1}"]
            for line in data:
                f.write(line + '\n')
                print(line)

    def create_predictions_for_queries(self, queries, object_for_calculation, bert):
        prediction_dict = dict()
        for index, row in queries.iterrows():
            qid = row['qid']
            query_str = row['text']
            query_as_df = pd.DataFrame(data=[query_str], columns=['text'])
            preprocessed_query = preprocessing.preprocess_data(query_as_df)
            retrieved_documents = object_for_calculation.retrieve_n_most_relevant_documents(preprocessed_query, 10)
            if bert:
                output_bert_df = self.bert.rerank_with_bert(retrieved_documents, query_str, 5)
                prediction_dict[qid] = output_bert_df['id'].values.tolist()
            else:
                retrieved_documents = retrieved_documents.head(5)
                prediction_dict[qid] = retrieved_documents['id'].values.tolist()

        return prediction_dict

    def evaluate_bm25(self):
        predictions = self.create_predictions_for_queries(self.negative_queries, self.negative_bm25, False)
        predictions.update(self.create_predictions_for_queries(self.neutral_queries, self.neutral_bm25, False))
        predictions.update(self.create_predictions_for_queries(self.positive_queries, self.positive_bm25, False))

        precision, recall, f1 = self.validate_F1k(self.ground_truth_dict_tfidf, predictions)
        self.write_to_evaluation_csv("evaluate_bm25", precision, recall, f1)

    def evaluate_bm25_with_bert(self):
        predictions = self.create_predictions_for_queries(self.negative_queries, self.negative_bm25, True)
        predictions.update(self.create_predictions_for_queries(self.neutral_queries, self.neutral_bm25, True))
        predictions.update(self.create_predictions_for_queries(self.positive_queries, self.positive_bm25, True))

        precision, recall, f1 = self.validate_F1k(self.ground_truth_dict_tfidf, predictions)
        self.write_to_evaluation_csv("evaluate_bm25_with_bert", precision, recall, f1)

    def evaluate_tfidf(self):
        predictions = self.create_predictions_for_queries(self.negative_queries, self.negative_tfidf, False)
        predictions.update(self.create_predictions_for_queries(self.neutral_queries, self.neutral_tfidf, False))
        predictions.update(self.create_predictions_for_queries(self.positive_queries, self.positive_tfidf, False))

        precision, recall, f1 = self.validate_F1k(self.ground_truth_dict_bm25, predictions)
        self.write_to_evaluation_csv("evaluate_tfidf", precision, recall, f1)

    def evaluate_tfidf_with_bert(self):
        predictions = self.create_predictions_for_queries(self.negative_queries, self.negative_tfidf, True)
        predictions.update(self.create_predictions_for_queries(self.neutral_queries, self.neutral_tfidf, True))
        predictions.update(self.create_predictions_for_queries(self.positive_queries, self.positive_tfidf, True))

        precision, recall, f1 = self.validate_F1k(self.ground_truth_dict_bm25, predictions)
        self.write_to_evaluation_csv("evaluate_tfidf_with_bert", precision, recall, f1)

    def get_precision_and_recall_k(self, prediction_docs: list[str], ground_truth_docs: list[str], k: int) -> (
    float, float):
        true_positives = 0
        false_positives = 0
        len_ground_truth_docs = len(ground_truth_docs)
        for doc in prediction_docs[0:k]:
            if doc in ground_truth_docs:
                true_positives += 1
            else:
                false_positives += 1
        if (true_positives + false_positives) == 0:
            precision = 0
        else:
            precision = true_positives / (true_positives + false_positives)
        if len_ground_truth_docs == 0:
            recall = 0
        else:
            recall = true_positives / len_ground_truth_docs
        return precision, recall

    def validate_F1k(self, ground_truth_dict, predictions: dict[str, list[str]], k: int = 5) -> float:
        total_f1 = 0
        total_precision = 0
        total_recall = 0
        for qid, prediction_docs in predictions.items():
            if qid in ground_truth_dict:
                docs_of_ground_truth = ground_truth_dict[qid]
                precision, recall = self.get_precision_and_recall_k(prediction_docs, docs_of_ground_truth, k)

                total_precision += precision
                total_recall += recall
                if (precision + recall) == 0:
                    total_f1 += 0
                else:
                    total_f1 += 2 * ((precision * recall) / (precision + recall))

        if len(predictions) == 0:
            average_f1 = 0
        else:
            average_f1 = total_f1 / len(predictions)

        average_precision = total_precision / len(predictions)
        average_recall = total_recall / len(predictions)

        return average_precision, average_recall, average_f1
