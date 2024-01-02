import os
import pandas as pd
from gensim import corpora, models, similarities

class BM25:
    def __init__(self, sentiments, documents, use_precomputed_bm25_model=False):
        # Reference: https://iamgeekydude.com/2022/12/25/bm25-using-python-gensim-package-search-engine-nlp/
        self.sentiments = sentiments
        self.preprocessed_documents = documents["preprocessed_text"].copy()
        self.raw_documents = documents[["text", "id", "sentiment"]].copy()
        self.temp_folder_name = "data/bm25_temp/"
        self.create_temp_folder()
        self.sim_matrices_file_name = self.temp_folder_name + '_'.join(self.sentiments) + '.simMatrices'
        self.dictionary_file_name = self.temp_folder_name + '_'.join(self.sentiments) + '.dictionary'
        self.raw_corpus_file_name = self.temp_folder_name + '_'.join(self.sentiments) + '.rawCorpus'

        # Hyperparameter
        self.k1 = 1.5
        self.b = 0.75

        if use_precomputed_bm25_model:
            self.dictionary = self.load_dictionary_from_file()
            self.raw_corpus = self.load_raw_corpus_from_file()
            self.bm25_model = models.OkapiBM25Model(dictionary=self.dictionary, k1=self.k1, b=self.b)
            self.tfidf_model = models.TfidfModel(dictionary=self.dictionary, smartirs='bnn')
            self.index = self.load_similarity_matrix_from_file()
        else:
            self.dictionary = self.create_dictionary_and_store_to_file()
            self.raw_corpus = self.create_raw_corpus_and_store_to_file()
            self.bm25_model = models.OkapiBM25Model(dictionary=self.dictionary, k1=self.k1, b=self.b)
            self.tfidf_model = models.TfidfModel(dictionary=self.dictionary, smartirs='bnn')
            self.index = self.compute_similarity_matrix_and_store_to_file()

    # region UTILS
    def create_temp_folder(self):
        if not os.path.exists(self.temp_folder_name):
            os.makedirs(self.temp_folder_name)
    # endregion

    # region Dictionary
    def create_dictionary_and_store_to_file(self):
        dictionary = corpora.Dictionary(self.preprocessed_documents)
        dictionary.save(self.dictionary_file_name)

        return dictionary

    def load_dictionary_from_file(self):
        dictionary = corpora.Dictionary.load(self.dictionary_file_name)

        return dictionary
    # endregion

    # region Raw-Corpus
    def create_raw_corpus_and_store_to_file(self):
        raw_corpus = [self.dictionary.doc2bow(current_document) for current_document in self.preprocessed_documents]
        corpora.MmCorpus.serialize(self.raw_corpus_file_name, raw_corpus)

        return raw_corpus

    def load_raw_corpus_from_file(self):
        raw_corpus = corpora.MmCorpus(self.raw_corpus_file_name)

        return raw_corpus
    # endregion

    # region Similarity-Matrix Computation
    def compute_similarity_matrix_and_store_to_file(self):
        # According to: https://iamgeekydude.com/2022/12/25/bm25-using-python-gensim-package-search-engine-nlp/
        # -> We disable normalization of the queries and documents!
        index = similarities.SparseMatrixSimilarity(self.bm25_model[self.raw_corpus], num_terms=len(self.dictionary), normalize_queries=False, normalize_documents=False)
        index.save(self.sim_matrices_file_name)

        return index

    def load_similarity_matrix_from_file(self):
        index = similarities.SparseMatrixSimilarity.load(self.sim_matrices_file_name)

        return index
    # endregion

    # region Document retrieval
    def retrieve_n_most_relevant_documents(self, preprocessed_query, n):
        # Representation of the query in TF-IDF
        query_tfidf_representation = self.tfidf_model[self.dictionary.doc2bow(preprocessed_query["preprocessed_text"].iloc[0])]

        # We retrieve the BM25-Scores of the query
        bm25_scores = self.index[query_tfidf_representation]
        bm25_scores_sorted = sorted(enumerate(bm25_scores), key=lambda x: -x[1])

        # We retrieve the indices and BM25-Scores of the relevant documents (of the dataframe!)
        relevant_indices = [index for index, similarity in bm25_scores_sorted[:n]]

        # We retrieve the corresponding documents according to the indices
        relevant_documents = []
        for i, current_index in enumerate(relevant_indices):
            current_document = self.raw_documents.iloc[current_index].copy()

            # We access the corresponding BM25-Score of the document
            similarity_score = bm25_scores_sorted[i][1]
            current_document['BM25-Score'] = similarity_score

            relevant_documents.append(current_document)

        # We convert the retrieved dictionary to a dataframe
        relevant_documents_dataframe = pd.DataFrame(relevant_documents)

        return relevant_documents_dataframe[['id', 'text', 'sentiment', 'BM25-Score']]
    # endregion