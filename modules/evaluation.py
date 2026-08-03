"""
evaluation.py

Information Retrieval Evaluation Metrics.

Implemented:

- Precision
- Recall
- F1 Score
- Precision@K
- Recall@K
- MAP
- MRR
- NDCG
"""


import math

import pandas as pd




class IREvaluator:


    def __init__(self):

        pass



# ------------------------------------------------------
# Precision
# ------------------------------------------------------


    def precision(
            self,
            retrieved,
            relevant
    ):
        """
        Precision:

        Relevant Retrieved /
        Total Retrieved

        """


        if len(retrieved) == 0:

            return 0



        relevant_retrieved = (

            set(retrieved)
            &
            set(relevant)

        )



        return round(

            len(relevant_retrieved)
            /
            len(retrieved),

            4

        )



# ------------------------------------------------------
# Recall
# ------------------------------------------------------


    def recall(
            self,
            retrieved,
            relevant
    ):
        """
        Recall:

        Relevant Retrieved /
        Total Relevant

        """


        if len(relevant)==0:

            return 0



        relevant_retrieved = (

            set(retrieved)
            &
            set(relevant)

        )



        return round(

            len(relevant_retrieved)
            /
            len(relevant),

            4

        )



# ------------------------------------------------------
# F1 Score
# ------------------------------------------------------


    def f1_score(
            self,
            precision,
            recall
    ):


        if precision + recall == 0:

            return 0



        return round(

            2 *
            (
                precision *
                recall
            )
            /
            (
                precision +
                recall
            ),

            4

        )



# ------------------------------------------------------
# Precision @ K
# ------------------------------------------------------


    def precision_at_k(
            self,
            ranked_results,
            relevant,
            k
    ):
        """
        Precision of top K results.
        """


        retrieved_k = ranked_results[:k]



        return self.precision(

            retrieved_k,

            relevant

        )



# ------------------------------------------------------
# Recall @ K
# ------------------------------------------------------


    def recall_at_k(
            self,
            ranked_results,
            relevant,
            k
    ):


        retrieved_k = ranked_results[:k]


        return self.recall(

            retrieved_k,

            relevant

        )



# ------------------------------------------------------
# Average Precision
# ------------------------------------------------------


    def average_precision(
            self,
            ranked_results,
            relevant
    ):
        """
        AP:

        Average precision after
        every relevant document.
        """


        score = 0

        hits = 0



        for i, doc in enumerate(
                ranked_results
        ):


            if doc in relevant:


                hits += 1


                score += hits/(i+1)



        if len(relevant)==0:

            return 0



        return round(

            score /
            len(relevant),

            4

        )



# ------------------------------------------------------
# Mean Average Precision
# ------------------------------------------------------


    def MAP(
            self,
            queries
    ):
        """
        queries format:


        [

        {
        retrieved:[...],
        relevant:[...]
        }

        ]

        """


        scores=[]



        for query in queries:


            scores.append(

                self.average_precision(

                    query["retrieved"],

                    query["relevant"]

                )

            )



        return round(

            sum(scores)
            /
            max(
                len(scores),
                1
            ),

            4

        )



# ------------------------------------------------------
# Mean Reciprocal Rank
# ------------------------------------------------------


    def MRR(
            self,
            ranked_results,
            relevant
    ):
        """
        Position of first
        relevant result.
        """


        for i, doc in enumerate(
                ranked_results
        ):


            if doc in relevant:


                return round(

                    1/(i+1),

                    4

                )


        return 0



# ------------------------------------------------------
# NDCG
# ------------------------------------------------------


    def NDCG(
            self,
            ranked_results,
            relevance_scores,
            k
    ):
        """
        Normalized Discounted
        Cumulative Gain.

        """


        def DCG(scores):

            value=0


            for i, score in enumerate(scores):


                value += (

                    score /
                    math.log2(
                        i+2
                    )

                )


            return value



        predicted_scores = [

            relevance_scores.get(
                doc,
                0
            )

            for doc in ranked_results[:k]

        ]



        ideal_scores = sorted(

            relevance_scores.values(),

            reverse=True

        )[:k]



        if DCG(ideal_scores)==0:

            return 0



        return round(

            DCG(predicted_scores)
            /
            DCG(ideal_scores),

            4

        )



# ------------------------------------------------------
# Complete Evaluation Report
# ------------------------------------------------------


    def evaluation_report(
            self,
            retrieved,
            relevant,
            k=10
    ):


        precision = self.precision(

            retrieved,

            relevant

        )


        recall = self.recall(

            retrieved,

            relevant

        )



        return {


            "Precision":

                precision,


            "Recall":

                recall,


            "F1":

                self.f1_score(

                    precision,

                    recall

                ),


            f"Precision@{k}":

                self.precision_at_k(

                    retrieved,

                    relevant,

                    k

                ),


            f"Recall@{k}":

                self.recall_at_k(

                    retrieved,

                    relevant,

                    k

                )

        }



# ------------------------------------------------------
# Convert Results To Table
# ------------------------------------------------------


    def comparison_table(
            self,
            results
    ):

        return pd.DataFrame(
            results
        )