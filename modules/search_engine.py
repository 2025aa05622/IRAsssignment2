"""
search_engine.py

Search engine module for
End-to-End Information Retrieval System.

Features:

- Keyword Search
- Boolean Search
- Phrase Search
- Fuzzy Search
- Ranked Retrieval
- Cosine Similarity
"""


import re

import numpy as np
import unicodedata

from sklearn.metrics.pairwise import cosine_similarity

from rapidfuzz import process, fuzz
import logging
logging.basicConfig(filename="app.log", level=logging.INFO)

class SearchEngine:


    def __init__(
            self,
            index_manager,
            processor,
            documents
    ):

        self.index = index_manager
        self.processor = processor
        self.documents = documents



# ------------------------------------------------------
# Basic Keyword Search
# ------------------------------------------------------

    def keyword_search(
            self,
            query
    ):
        """
        Retrieve documents containing
        query terms.
        """

        originalquery = query
        text = self.processor.clean_text(originalquery)
        tokens = self.processor.tokenize(text)
        #terms = query.lower().split()
        terms = self.processor.preprocess(query)

        documents = set()

        for term in terms:
            posting = (
                self.index
                .get_posting_list(term)
            )
            documents.update(
                posting
            )

        results = []
        retrieved_results = []

        
        for doc_id in documents:
            doc = self.documents[doc_id - 1]
            content = doc["content"].lower()
            matched_terms = 0

            for term in tokens:

                if re.search(r"\b" + re.escape(term) + r"\b", content):
                    matched_terms += 1
            

            if matched_terms > 0:

                results.append({
                    "document_id": doc_id,
                    "score": matched_terms / len(terms)
                })

            retrieved_results.append({
                "ret_document_id": doc_id
            })

        return results, retrieved_results



# ------------------------------------------------------
# Boolean AND Search
# ------------------------------------------------------


    def boolean_and_search(
            self,
            query
    ):
        """
        Example:

        artificial AND intelligence

        returns documents
        containing both terms
        """


        terms = (

            query
            .lower()
            .replace(
                "AND",
                ""
            )
            .split()

        )


        result = None



        for term in terms:


            docs = set(

                self.index
                .get_posting_list(term)

            )


            if result is None:

                result = docs


            else:

                result = result.intersection(
                    docs
                )



        return list(result)



# ------------------------------------------------------
# Boolean OR Search
# ------------------------------------------------------


    def boolean_or_search(
            self,
            query
    ):
        """
        Example:

        AI OR ML
        """


        terms = (

            query
            .lower()
            .replace(
                "OR",
                ""
            )
            .split()

        )


        result = set()



        for term in terms:


            result.update(

                self.index
                .get_posting_list(term)

            )



        return list(result)


# ------------------------------------------------------
# Phrase Search
# ------------------------------------------------------


    def phrase_search(
            self,
            phrase
    ):
        """
        Search exact phrase occurrence.

        Example:

        "artificial intelligence"

        """        

        originalphrase = phrase

        phrase = (
            phrase
            .lower()
            .replace(
                '"',
                ""
            )
        )

        phrase = self.processor.tokens_to_text(self.processor.preprocess(phrase))


        matching_documents = []



        for doc_id, text in self.index.documents.items():


            content = text.lower()



            if phrase in content:


                matching_documents.append(
                    doc_id
                )        

        results = []        
        retrieved_results = []

        pattern1 = originalphrase.lower()
        pattern1 = pattern1.replace("*", ".*")
        pattern1 = pattern1.replace("$", ".")

        regex1 = re.compile("^" + pattern1 + "$")

        for doc_id in matching_documents:
            doc = self.documents[doc_id - 1]
            newphrase = originalphrase.replace("’", "'").replace("‘", "'").replace("`", "'")
            content =doc["content"].replace("’", "'").replace("‘", "'").replace("`", "'")
            if newphrase in content:
                results.append({
                    "document_id": doc_id,
                    "score": 1.0
                })            
                
            retrieved_results.append({
                "ret_document_id": doc_id
            })

        return results, retrieved_results
        #return matching_documents



# ------------------------------------------------------
# Wildcard Search
# ------------------------------------------------------


    def wildcard_search(
            self,
            pattern
    ):
        """
        Supports wildcard queries.

        Example:

        comput*

        matches:

        computer
        computing
        computational

        """

        origpattrn = pattern;
        phrase = pattern.replace("*", "")
        phrase = self.processor.tokens_to_text(self.processor.preprocess(phrase))
        phrase += '*'
        pattern = phrase

        regex_pattern = (
            pattern
            .lower()
            .replace(
                "*",
                ".*"
            )

        )


        regex = re.compile(
            "^" + regex_pattern + "$"
        )



        matched_terms = []



        for term in self.index.get_vocabulary():
            if regex.match(term):
                matched_terms.append(
                    term
                )
        documents = set()



        for term in matched_terms:
            documents.update(
                self.index
                .get_posting_list(term)

            )

        results = []
        retrieved_results = []

        pattern1 = origpattrn.lower()
        pattern1 = pattern1.replace("*", ".*")
        pattern1 = pattern1.replace("$", ".")

        regex1 = re.compile("^" + pattern1 + "$")
        
        for doc_id in documents:

            doc = self.documents[doc_id - 1]
            content = doc["content"].lower()

            words = re.findall(r"\b\w+\b", content)

            if any(regex1.match(word) for word in words):
                results.append({
                    "document_id": doc_id,
                    "score": 1.0
                })

            retrieved_results.append({
                 "ret_document_id": doc_id
            })

        return results, retrieved_results



# ------------------------------------------------------
# Fuzzy Search
# ------------------------------------------------------


    def fuzzy_search(
            self,
            query,
            threshold=80
    ):
        """
        Correct spelling mistakes.

        Example:

        intellgence

        becomes:

        intelligence

        """


        query = self.processor.tokens_to_text(self.processor.preprocess(query))

        vocabulary = (
            self.index
            .get_vocabulary()
        )



        matches = process.extract(

            query,

            vocabulary,

            scorer=fuzz.ratio,

            limit=5

        )



        valid_terms = []

        for term, score, _ in matches:
            if score >= threshold:
                valid_terms.append(
                    (
                        term,
                        score
                    )

                )

        documents = set()

        for term, score in valid_terms:

            term1 = self.processor.tokens_to_text(self.processor.preprocess(term))
            documents.update(

                self.index
                .get_posting_list(term1)

            )

        results = []
        retrieved_results = []
            
        for doc_id in documents:
            results.append({
                "document_id": doc_id,
                "score": 1.0
            })

            retrieved_results.append({
                 "ret_document_id": doc_id
            })

        return results, retrieved_results



# ------------------------------------------------------
# Query Normalization
# ------------------------------------------------------


    def normalize_query(
            self,
            query
    ):
        """
        Clean user query.
        """


        query = query.lower()


        query = re.sub(

            r"[^\w\s]",

            " ",

            query

        )


        query = " ".join(
            query.split()
        )


        return query


# ------------------------------------------------------
# Ranked Retrieval using TF-IDF
# ------------------------------------------------------


    def ranked_search(
        self,
        query,
        top_k=10
    ):
        """
        Ranked retrieval using:

        TF-IDF
        +
        Cosine Similarity

        Query and documents use the same
        preprocessing pipeline.
        """

        originalquery = query
        text = self.processor.clean_text(originalquery)
        tokens = self.processor.tokenize(text)
        
        if self.index.tfidf_matrix is None:
            raise Exception(
                "TF-IDF index not available"
            )


        # ---------------------------------
        # 1. Preprocess query
        # ---------------------------------

        processed_query = (
            self.processor.tokens_to_text(
                    self.processor.preprocess(query)
                )
        )


        # Convert list back to text
        # because sklearn TF-IDF expects string

        # processed_query = " ".join(
        #     processed_query
        # )


        if not processed_query.strip():

            return []


        # ---------------------------------
        # 2. Convert query into TF-IDF vector
        # ---------------------------------

        query_vector = (
            self.index.vectorizer
            .transform(
                [processed_query]
            )
        )


        # ---------------------------------
        # 3. Calculate cosine similarity
        # ---------------------------------

        scores = cosine_similarity(
            query_vector,
            self.index.tfidf_matrix
        )[0]


        # ---------------------------------
        # 4. Map scores to documents
        # ---------------------------------

        document_ids = list(
            self.index.documents.keys()
        )


        ranked_results = []
        retrieved_results = []


        for doc_id, score in zip(
                document_ids,
                scores
        ):

            # ignore unrelated documents

            if score > 0:
                
                doc = self.documents[doc_id - 1]
                content = doc["content"].lower()
                matched_terms = 0

                for term in tokens:

                    if re.search(r"\b" + re.escape(term) + r"\b", content):
                        matched_terms += 1
                

                if matched_terms > 0:
                    ranked_results.append(
                        {
                            "document_id": doc_id,

                            "score": round(
                                float(score),
                                4
                            )
                        }
                    )

                retrieved_results.append({
                    "ret_document_id": doc_id
                })


        # ---------------------------------
        # 5. Rank by highest similarity
        # ---------------------------------

        ranked_results.sort(
            key=lambda x: x["score"],
            reverse=True
        )


        return ranked_results[:top_k], retrieved_results



# ------------------------------------------------------
# Search With Multiple Strategies
# ------------------------------------------------------


    def search(
            self,
            query,
            mode="ranked",
            top_k=10
    ):
        """
        Unified search API.

        Modes:

        ranked
        keyword
        phrase
        fuzzy
        wildcard

        """


        if mode == "ranked":


            return self.ranked_search(

                query,

                top_k

            )


        elif mode == "keyword":


            return self.keyword_search(
                query
            )


        elif mode == "phrase":


            return self.phrase_search(
                query
            )


        elif mode == "fuzzy":


            return self.fuzzy_search(
                query
            )


        elif mode == "wildcard":


            return self.wildcard_search(
                query
            )


        else:

            raise ValueError(

                "Invalid search mode"

            )



# ------------------------------------------------------
# Highlight Query Terms
# ------------------------------------------------------


    def highlight_terms(
            self,
            text,
            query
    ):
        """
        Highlight matching terms
        for UI display.
        """


        terms = query.split()

        highlighted = text

        for term in terms:
            highlighted = re.sub(

                f"({term})",

                r"**\1**",

                highlighted,

                flags=re.IGNORECASE

            )



        return highlighted



# ------------------------------------------------------
# Search Statistics
# ------------------------------------------------------


    def search_statistics(
            self,
            results
    ):
        """
        Generate analytics
        for search results.
        """


        scores = [

            r["score"]

            for r in results

            if "score" in r

        ]



        if not scores:

            return {

                "results":0,

                "average_score":0

            }



        return {


            "results":

                len(results),


            "average_score":

                round(

                    sum(scores)
                    /
                    len(scores),

                    4

                ),


            "highest_score":

                max(scores)

        }