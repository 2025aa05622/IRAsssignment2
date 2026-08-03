"""
preprocessing.py

Text preprocessing and mining framework for
End-to-End Information Retrieval System.

Features:
- Cleaning
- Tokenization
- Stopword removal
- Stemming
- Lemmatization
- Keyword extraction
- Document profiling
- Feature statistics
"""

import re
import string

from collections import Counter

import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer
)


# ------------------------------------------------------
# Download required NLP resources
# ------------------------------------------------------

def download_nltk_resources():

    resources = [
        "punkt",
        "stopwords",
        "wordnet",
        "omw-1.4"
    ]

    for resource in resources:

        try:

            nltk.data.find(resource)

        except LookupError:

            nltk.download(resource)



download_nltk_resources()


# ------------------------------------------------------
# Text Preprocessor Class
# ------------------------------------------------------

class TextPreprocessor:


    def __init__(
            self,
            remove_stopwords=True,
            stemming=True,
            lemmatization=True
    ):

        self.remove_stopwords = remove_stopwords

        self.enable_stemming = stemming

        self.enable_lemmatization = lemmatization


        self.stop_words = set(
            stopwords.words("english")
        )


        self.stemmer = PorterStemmer()


        self.lemmatizer = WordNetLemmatizer()



    # --------------------------------------------------
    # Remove HTML
    # --------------------------------------------------

    def remove_html(
            self,
            text
    ):

        text = re.sub(
            r"<.*?>",
            " ",
            text
        )

        return text



    # --------------------------------------------------
    # Lowercase Conversion
    # --------------------------------------------------

    def lowercase(
            self,
            text
    ):

        return text.lower()



    # --------------------------------------------------
    # Remove URLs
    # --------------------------------------------------

    def remove_urls(
            self,
            text
    ):

        return re.sub(

            r"http\S+|www\S+",

            " ",

            text

        )



    # --------------------------------------------------
    # Remove Punctuation
    # --------------------------------------------------

    def remove_punctuation(
            self,
            text
    ):


        return text.translate(

            str.maketrans(

                "",
                "",

                string.punctuation

            )

        )



    # --------------------------------------------------
    # Remove Numbers
    # --------------------------------------------------

    def remove_numbers(
            self,
            text
    ):

        return re.sub(

            r"\d+",

            " ",

            text

        )



    # --------------------------------------------------
    # Remove Extra Spaces
    # --------------------------------------------------

    def normalize_spaces(
            self,
            text
    ):

        return " ".join(
            text.split()
        )



    # --------------------------------------------------
    # Basic Cleaning Pipeline
    # --------------------------------------------------

    def clean_text(
            self,
            text
    ):


        text = self.remove_html(text)


        text = self.remove_urls(text)


        text = self.lowercase(text)


        text = self.remove_punctuation(text)


        text = self.remove_numbers(text)


        text = self.normalize_spaces(text)


        return text

    # --------------------------------------------------
    # Tokenization
    # --------------------------------------------------

    def tokenize(
            self,
            text
    ):
        """
        Convert text into individual tokens.
        """

        tokens = nltk.word_tokenize(text)

        return tokens



    # --------------------------------------------------
    # Remove Stopwords
    # --------------------------------------------------

    def remove_stop_words(
            self,
            tokens
    ):
        """
        Remove common English words
        like:
        the, is, are, and etc.
        """

        filtered = []

        for token in tokens:

            if token not in self.stop_words:

                filtered.append(token)


        return filtered



    # --------------------------------------------------
    # Stemming
    # --------------------------------------------------

    def stem_words(
            self,
            tokens
    ):
        """
        Convert words into root forms.

        Example:

        playing
        played
        plays

        becomes:

        play
        """

        stemmed = []


        for token in tokens:

            stemmed.append(

                self.stemmer.stem(token)

            )


        return stemmed



    # --------------------------------------------------
    # Lemmatization
    # --------------------------------------------------

    def lemmatize_words(
            self,
            tokens
    ):
        """
        Convert words into meaningful base forms.

        Example:

        running -> run
        better -> good
        """

        lemmatized = []


        for token in tokens:

            lemmatized.append(

                self.lemmatizer.lemmatize(token)

            )


        return lemmatized



    # --------------------------------------------------
    # Complete NLP Pipeline
    # --------------------------------------------------

    def preprocess(
            self,
            text
    ):
        """
        Complete preprocessing pipeline.

        Raw text
            |
            v
        Cleaning
            |
            v
        Tokenization
            |
            v
        Stopword Removal
            |
            v
        Stemming
            |
            v
        Lemmatization
        """


        # Cleaning

        text = self.clean_text(text)



        # Tokenization

        tokens = self.tokenize(text)



        # Stopword removal

        if self.remove_stopwords:

            tokens = self.remove_stop_words(
                tokens
            )



        # Stemming

        if self.enable_stemming:

            tokens = self.stem_words(
                tokens
            )



        # Lemmatization

        if self.enable_lemmatization:

            tokens = self.lemmatize_words(
                tokens
            )



        return tokens



    # --------------------------------------------------
    # Convert Tokens Back To Text
    # --------------------------------------------------

    def tokens_to_text(
            self,
            tokens
    ):

        return " ".join(tokens)


    # --------------------------------------------------
    # TF-IDF Feature Extraction
    # --------------------------------------------------

    def extract_tfidf_features(
            self,
            documents,
            max_features=5000
    ):
        """
        Convert documents into TF-IDF vectors.

        Input:
            documents:
                List of document strings

        Output:
            TF-IDF matrix
            Feature names
        """


        vectorizer = TfidfVectorizer(

            max_features=max_features

        )


        tfidf_matrix = vectorizer.fit_transform(
            documents
        )


        feature_names = (
            vectorizer
            .get_feature_names_out()
        )


        return {

            "matrix": tfidf_matrix,

            "features": feature_names,

            "vectorizer": vectorizer

        }



    # --------------------------------------------------
    # Count Vector Features
    # --------------------------------------------------

    def extract_count_features(
            self,
            documents,
            max_features=5000
    ):
        """
        Bag-of-Words feature extraction.
        """


        vectorizer = CountVectorizer(

            max_features=max_features

        )


        count_matrix = vectorizer.fit_transform(
            documents
        )


        features = (
            vectorizer
            .get_feature_names_out()
        )


        return {

            "matrix": count_matrix,

            "features": features,

            "vectorizer": vectorizer

        }



    # --------------------------------------------------
    # Keyword Extraction
    # --------------------------------------------------

    def extract_keywords(
            self,
            document,
            top_n=10
    ):
        """
        Extract important keywords from
        a single document using TF-IDF.
        """


        result = self.extract_tfidf_features(

            [document]

        )


        scores = result["matrix"].toarray()[0]


        words = result["features"]



        keyword_scores = list(

            zip(
                words,
                scores
            )

        )


        keyword_scores.sort(

            key=lambda x: x[1],

            reverse=True

        )


        return keyword_scores[:top_n]



    # --------------------------------------------------
    # Word Frequency Analysis
    # --------------------------------------------------

    def word_frequency(
            self,
            documents,
            top_n=20
    ):
        """
        Find most frequent words
        in the corpus.
        """


        all_tokens = []


        for doc in documents:


            tokens = self.preprocess(doc)


            all_tokens.extend(tokens)



        frequency = Counter(
            all_tokens
        )


        return frequency.most_common(
            top_n
        )



    # --------------------------------------------------
    # Document Profile
    # --------------------------------------------------

    def document_profile(
            self,
            document
    ):
        """
        Generate statistics for a document.
        """


        tokens = self.preprocess(
            document
        )


        profile = {


            "characters":
                len(document),


            "words":
                len(document.split()),


            "unique_words":
                len(set(tokens)),


            "vocabulary_ratio":
                round(
                    len(set(tokens))
                    /
                    max(len(tokens),1),
                    3
                ),


            "top_terms":
                Counter(tokens)
                .most_common(10)

        }


        return profile



    # --------------------------------------------------
    # Corpus Statistics
    # --------------------------------------------------

    def corpus_statistics(
            self,
            documents
    ):
        """
        Analyze complete corpus.
        """


        total_documents = len(
            documents
        )


        total_words = 0


        vocabulary = set()


        document_lengths = []



        for doc in documents:


            tokens = self.preprocess(
                doc
            )


            total_words += len(tokens)


            vocabulary.update(
                tokens
            )


            document_lengths.append(
                len(tokens)
            )



        statistics = {


            "documents":
                total_documents,


            "total_words":
                total_words,


            "vocabulary_size":
                len(vocabulary),


            "average_document_length":
                round(

                    sum(document_lengths)
                    /
                    max(
                        len(document_lengths),
                        1
                    ),

                    2
                )


        }


        return statistics


    # --------------------------------------------------
    # Generate N-Grams
    # --------------------------------------------------

    def generate_ngrams(
            self,
            tokens,
            n=2
    ):
        """
        Generate n-grams.

        Example:

        Tokens:
        ["machine","learning","model"]

        Bigram:

        machine_learning
        learning_model
        """


        ngrams = []


        for i in range(
            len(tokens)-n+1
        ):

            gram = "_".join(

                tokens[i:i+n]

            )

            ngrams.append(
                gram
            )


        return ngrams



    # --------------------------------------------------
    # Preprocessing Strategy Comparison
    # --------------------------------------------------

    def compare_preprocessing(
            self,
            document
    ):
        """
        Compare different preprocessing
        approaches.

        Useful for assignment analysis.
        """


        result = {}



        # Only cleaning

        cleaned = self.clean_text(
            document
        )


        result["clean_only"] = {


            "tokens":
                len(cleaned.split()),


            "unique_terms":
                len(
                    set(
                        cleaned.split()
                    )
                )

        }



        # Stopword removal

        tokens = self.tokenize(
            cleaned
        )


        no_stopwords = self.remove_stop_words(
            tokens
        )


        result["stopword_removed"] = {


            "tokens":
                len(no_stopwords),


            "unique_terms":
                len(
                    set(
                        no_stopwords
                    )
                )

        }



        # Stemming

        stemmed = self.stem_words(
            no_stopwords
        )


        result["stemming"] = {


            "tokens":
                len(stemmed),


            "unique_terms":
                len(
                    set(
                        stemmed
                    )
                )

        }



        # Lemmatization

        lemma = self.lemmatize_words(
            no_stopwords
        )


        result["lemmatization"] = {


            "tokens":
                len(lemma),


            "unique_terms":
                len(
                    set(
                        lemma
                    )
                )

        }



        return result



    # --------------------------------------------------
    # Process Complete Corpus
    # --------------------------------------------------

    def process_corpus(
            self,
            documents
    ):
        """
        Apply preprocessing on
        multiple documents.
        """


        processed_documents = []


        for doc in documents:


            tokens = self.preprocess(
                doc
            )


            processed_documents.append(

                self.tokens_to_text(
                    tokens
                )

            )


        return processed_documents



    # --------------------------------------------------
    # Build Vocabulary
    # --------------------------------------------------

    def build_vocabulary(
            self,
            documents
    ):
        """
        Create corpus vocabulary.
        """


        vocabulary = set()



        for doc in documents:


            tokens = self.preprocess(
                doc
            )


            vocabulary.update(
                tokens
            )


        return vocabulary



    # --------------------------------------------------
    # Vocabulary Comparison
    # --------------------------------------------------

    def vocabulary_comparison(
            self,
            documents
    ):
        """
        Compare vocabulary size
        before and after preprocessing.
        """


        original_vocab = set()


        processed_vocab = set()



        for doc in documents:


            words = doc.lower().split()


            original_vocab.update(
                words
            )



            tokens = self.preprocess(
                doc
            )


            processed_vocab.update(
                tokens
            )



        return {


            "original_vocabulary":

                len(original_vocab),


            "processed_vocabulary":

                len(processed_vocab),


            "reduction_percentage":

                round(

                    (
                        1 -
                        (
                            len(processed_vocab)
                            /
                            max(
                                len(original_vocab),
                                1
                            )
                        )

                    )
                    *
                    100,

                    2
                )

        }



    # --------------------------------------------------
    # Export Processed Corpus
    # --------------------------------------------------

    def export_processed_documents(
            self,
            documents,
            filename="processed_documents.txt"
    ):
        """
        Save processed documents.
        """


        processed = self.process_corpus(
            documents
        )


        with open(
            filename,
            "w",
            encoding="utf-8"
        ) as file:


            for doc in processed:


                file.write(
                    doc
                    +
                    "\n\n"
                )


        return filename