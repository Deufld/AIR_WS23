import pandas as pd
import scipy
from sentence_transformers import SentenceTransformer


class BERT:
    def __init__(self):
        self.embedder = SentenceTransformer('bert-base-nli-mean-tokens')

    def rerank_with_bert(self, retrieved_documents, query_str, n) -> pd.DataFrame:
        # Taken from: https://medium.com/@papai143/information-retrieval-with-document-re-ranking-with-bert-and-bm25-7c29d738df73
        corpus_embeddings = self.embedder.encode(retrieved_documents['text'].values.tolist())
        queries = [query_str]
        query_embeddings = self.embedder.encode(queries)
        closest_n = n

        output_df = pd.DataFrame(columns=['id', 'text', 'score'])

        for query, query_embedding in zip(queries, query_embeddings):
            distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

            results = zip(range(len(distances)), distances)
            results = sorted(results, key=lambda x: x[1])

            for idx, distance in results[0:closest_n]:
                output_df.loc[len(output_df.index)] = [retrieved_documents['id'].iloc[idx], retrieved_documents['text'].iloc[idx].strip(), 1 - distance]

        return output_df
