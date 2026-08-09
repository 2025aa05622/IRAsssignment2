"""
indexing.py

Builds and manages indexes for the
End-to-End Information Retrieval System.

Implemented:

1. Inverted Index
2. TF-IDF Index
3. Posting Lists
4. Index Statistics
5. Index Persistence
"""

import math
import pickle
from collections import defaultdict
import nltk
import logging
# logging.basicConfig(filename="app.log", level=logging.INFO)

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


from config import INDEX_DIR
from sklearn.feature_extraction.text import TfidfVectorizer


class IndexManager:

    def __init__(self):
        self.inverted_index = defaultdict(list)
        self.document_lengths = {}


        self.document_frequency = {}


        self.documents = {}

        self.tfidf_matrix = None

        self.vectorizer = None

    # ------------------------------------------------------
    # Add Document
    # ------------------------------------------------------

    def add_document(self, document_id, text):
        """
        Add document into memory.
        """

        self.documents[document_id] = text

    # ------------------------------------------------------
    # Tokenize
    # ------------------------------------------------------

    def tokenize(self, text):

        return text.lower().split()

    # ------------------------------------------------------
    # Build Inverted Index
    # ------------------------------------------------------

    def build_inverted_index(self, documents):
        self.inverted_index = defaultdict(list)
        self.document_frequency = {}
        self.document_lengths = {}
        """
        Create:

        term
          |
          |
          +---- document ids


        Example:

        artificial

              |
              |
          [1,5,8]


        """

        for doc in documents:

            doc_id = doc["document_id"]

            content = doc["content"]

            self.documents[doc_id] = content

            tokens = self.tokenize(content)

            self.document_lengths[doc_id] = len(tokens)

            unique_terms = set(tokens)
            # logging.info(unique_terms)

            for term in unique_terms:

                self.inverted_index[term].append(doc_id)

        return self.inverted_index

    # ------------------------------------------------------
    # Document Frequency
    # ------------------------------------------------------

    def calculate_document_frequency(self):

        for term, docs in self.inverted_index.items():

            self.document_frequency[term] = len(docs)

        return self.document_frequency

    # ------------------------------------------------------
    # TF Calculation
    # ------------------------------------------------------

    def term_frequency(self, term, document):

        tokens = self.tokenize(document)

        count = tokens.count(term)

        return count / max(len(tokens), 1)

    # ------------------------------------------------------
    # IDF Calculation
    # ------------------------------------------------------

    def inverse_document_frequency(self, term):

        total_docs = len(self.documents)

        doc_count = self.document_frequency.get(term, 0)

        if doc_count == 0:

            return 0

        return math.log(total_docs / doc_count)

    # ------------------------------------------------------
    # TF-IDF Score
    # ------------------------------------------------------

    def tfidf_score(self, term, document):

        tf = self.term_frequency(term, document)

        idf = self.inverse_document_frequency(term)

        return tf * idf

    # ------------------------------------------------------
    # Build TF-IDF Matrix
    # ------------------------------------------------------

    def build_tfidf_matrix(self, documents):
        """
        Create TF-IDF vector representation.

        Used for:
        - Ranked search
        - Similarity calculation
        - Recommendation
        """

        ids = []

        texts = []

        for doc in documents:

            ids.append(doc["document_id"])

            texts.append(doc["content"])

        self.vectorizer = TfidfVectorizer()

        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

        return {
            "document_ids": ids,
            "matrix": self.tfidf_matrix,
            "features": self.vectorizer.get_feature_names_out(),
        }

    # ------------------------------------------------------
    # Get Vector Representation
    # ------------------------------------------------------

    def transform_query(self, query):
        """
        Convert user query into
        TF-IDF vector.
        """

        if self.vectorizer is None:

            raise Exception("TF-IDF index not built")

        return self.vectorizer.transform([query])

    # ------------------------------------------------------
    # Get Posting List
    # ------------------------------------------------------

    def get_posting_list(self, term):
        """
        Return documents containing a term.
        """

        return self.inverted_index.get(term, [])

    # ------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------

    def get_vocabulary(self):

        return list(self.inverted_index.keys())

    # ------------------------------------------------------
    # Index Statistics
    # ------------------------------------------------------

    def index_statistics(self):
        """
        Generate index analytics.
        """

        total_terms = len(self.inverted_index)

        total_documents = len(self.documents)

        total_postings = sum(len(value) for value in self.inverted_index.values())

        average_document_length = 0

        if total_documents > 0:

            average_document_length = (
                sum(self.document_lengths.values()) / total_documents
            )

        return {
            "documents": total_documents,
            "unique_terms": total_terms,
            "total_postings": total_postings,
            "average_document_length": round(average_document_length, 2),
        }

    # ------------------------------------------------------
    # Save Index
    # ------------------------------------------------------

    def save_index(self, filename="inverted_index.pkl"):
        """
        Persist index to disk.
        """

        path = INDEX_DIR / filename

        data = {
            "inverted_index": dict(self.inverted_index),
            "document_frequency": self.document_frequency,
            "document_lengths": self.document_lengths,
            "documents": self.documents,
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
        }

        with open(path, "wb") as file:

            pickle.dump(data, file)

        return str(path)

    # ------------------------------------------------------
    # Load Index
    # ------------------------------------------------------

    def load_index(self, filename="inverted_index.pkl"):
        """
        Load saved index.
        """

        path = INDEX_DIR / filename

        with open(path, "rb") as file:

            data = pickle.load(file)

        self.inverted_index = defaultdict(list, data["inverted_index"])

        self.document_frequency = data["document_frequency"]

        self.document_lengths = data["document_lengths"]

        self.documents = data["documents"]

        self.vectorizer = data["vectorizer"]

        self.tfidf_matrix = data["tfidf_matrix"]

        return True
