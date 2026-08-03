"""
recommender.py

Content Based Recommendation System.

Uses:

TF-IDF
+
Cosine Similarity

to recommend similar documents.
"""


import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity




class ContentBasedRecommender:


    def __init__(self):

        self.vectorizer = None

        self.tfidf_matrix = None

        self.documents = None

        self.document_ids = []



# ------------------------------------------------------
# Build Recommendation Model
# ------------------------------------------------------


    def fit(
            self,
            documents
    ):
        """
        Train recommender.

        documents format:

        [
        {
        "document_id":1,
        "content":"text..."
        }
        ]

        """


        self.documents = documents



        self.document_ids = [

            doc["document_id"]

            for doc in documents

        ]



        texts = [

            doc["content"]

            for doc in documents

        ]



        self.vectorizer = TfidfVectorizer()



        self.tfidf_matrix = (

            self.vectorizer
            .fit_transform(texts)

        )



        return True



# ------------------------------------------------------
# Recommend Similar Documents
# ------------------------------------------------------


    def recommend(
            self,
            document_id,
            top_k=5
    ):
        """
        Return similar documents.

        """


        if self.tfidf_matrix is None:

            raise Exception(
                "Model not trained"
            )



        index = (

            self.document_ids
            .index(
                document_id
            )

        )



        document_vector = (

            self.tfidf_matrix[index]

        )



        similarity_scores = (

            cosine_similarity(

                document_vector,

                self.tfidf_matrix

            )[0]

        )



        results = []



        for i, score in enumerate(
                similarity_scores
        ):



            if self.document_ids[i] != document_id:


                results.append(

                    {

                    "document_id":

                        self.document_ids[i],


                    "similarity":

                        round(
                            float(score),
                            4
                        )

                    }

                )



        results.sort(

            key=lambda x:
            x["similarity"],

            reverse=True

        )



        return results[:top_k]



# ------------------------------------------------------
# Recommend By Text Query
# ------------------------------------------------------


    def recommend_by_text(
            self,
            text,
            top_k=5
    ):
        """
        Recommend documents based
        on user input text.
        """


        query_vector = (

            self.vectorizer
            .transform(
                [text]
            )

        )



        scores = (

            cosine_similarity(

                query_vector,

                self.tfidf_matrix

            )[0]

        )



        results = []



        for i, score in enumerate(scores):


            results.append(

                {

                "document_id":

                    self.document_ids[i],


                "similarity":

                    round(
                        float(score),
                        4
                    )

                }

            )



        results.sort(

            key=lambda x:
            x["similarity"],

            reverse=True

        )


        return results[:top_k]



# ------------------------------------------------------
# Similarity Matrix
# ------------------------------------------------------


    def similarity_matrix(self):

        """
        Return complete document
        similarity matrix.
        """


        return cosine_similarity(

            self.tfidf_matrix

        )



# ------------------------------------------------------
# Recommendation Analytics
# ------------------------------------------------------


    def recommendation_statistics(
            self,
            recommendations
    ):


        scores = [

            r["similarity"]

            for r in recommendations

        ]



        return {


            "recommendations":

                len(recommendations),


            "average_similarity":

                round(

                    sum(scores)
                    /
                    max(
                        len(scores),
                        1
                    ),

                    4

                ),


            "highest_similarity":

                max(scores)

                if scores

                else 0

        }