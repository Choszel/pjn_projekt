import spacy
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class Retriever:
    def __init__(self, corpus: List[Dict], nlp_model=None, algorithm: str = "BM25"):
        self.corpus = corpus        
        self.selected_algorithm = algorithm
        if nlp_model:
            self.nlp = nlp_model
        else:
            try:
                self.nlp = spacy.load("pl_core_news_sm")
            except OSError:
                raise ImportError("Model 'pl_core_news_sm' nie jest zainstalowany.")
        
        match algorithm:
            case "BM25":
                self.tokenized_corpus = [doc["lemmatized"].split() for doc in self.corpus]
                self.bm25 = BM25Okapi(self.tokenized_corpus)
            case "SentenceBERT":
                self.sentenceBERT = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
                self.documents = [doc["content"] for doc in self.corpus]
                self.doc_embeddings = self.sentenceBERT.encode(
                    self.documents,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            case "TFIDF":
                self.documents = [doc["lemmatized"] for doc in self.corpus]

                self.vectorizer = TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9
                )
                self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            case _:
                raise ValueError(f"Nieznany algorytm: {algorithm}")
            
    def preprocess_query(self, query: str) -> List[str]:

        doc = self.nlp(query.lower())

        processed_tokens = [
            token.lemma_ for token in doc
            if not token.is_punct and not token.is_space and not token.is_stop
        ]
        return processed_tokens

    def retrieve_top_k(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:

        match self.selected_algorithm:
            case "BM25":
                tokenized_query = self.preprocess_query(query)
                if not tokenized_query:
                    return []
                doc_scores = self.bm25.get_scores(tokenized_query)

            case "SentenceBERT":
                query_embedding = self.sentenceBERT.encode(
                    query,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                doc_scores = cosine_similarity(
                    [query_embedding],
                    self.doc_embeddings
                )[0]
                
            case "TFIDF":
                processed_query = self.preprocess_query(query)
                if not processed_query:
                    return []
                processed_query = " ".join(processed_query)

                query_vector = self.vectorizer.transform([processed_query])

                doc_scores = cosine_similarity(
                    query_vector,
                    self.tfidf_matrix
                )[0]
                
            case _:
                raise ValueError(f"Nieznany algorytm: {self.selected_algorithm}")

        results = sorted(
            zip(self.corpus, doc_scores),
            key=lambda x: x[1],
            reverse=True
        )

        return results[:k]
