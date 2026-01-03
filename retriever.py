import spacy
from rank_bm25 import BM25Okapi
from typing import List, Tuple, Dict

class Retriever:

    def __init__(self, corpus: List[Dict], nlp_model=None):
       self.corpus = corpus

       if nlp_model:
            self.nlp = nlp_model
       else:
            try:
                self.nlp = spacy.load("pl_core_news_sm")
            except OSError:
                raise ImportError("Model 'pl_core_news_sm' nie jest zainstalowany.")

       self.tokenized_corpus = [
            doc["lemmatized"].split() for doc in self.corpus
        ]

       self.bm25 = BM25Okapi(self.tokenized_corpus)

    def preprocess_query(self, query: str) -> List[str]:

        doc = self.nlp(query.lower())

        processed_tokens = [
            token.lemma_ for token in doc
            if not token.is_punct and not token.is_space and not token.is_stop
        ]
        return processed_tokens

    def retrieve_top_k(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:

        tokenized_query = self.preprocess_query(query)

        if not tokenized_query:
            return []

        doc_scores = self.bm25.get_scores(tokenized_query)

        results = sorted(
            zip(self.corpus, doc_scores),
            key=lambda x: x[1],
            reverse=True
        )

        return results[:k]
