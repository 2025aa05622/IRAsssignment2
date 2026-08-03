"""
visualization.py

Visualization utilities for
Information Retrieval System.

Used by Streamlit dashboard.

"""

import matplotlib.pyplot as plt

import pandas as pd



class IRVisualizer:


    def __init__(self):

        pass



# ------------------------------------------------------
# Word Frequency Chart
# ------------------------------------------------------


    def word_frequency_chart(
            self,
            frequency_data,
            top_n=20
    ):
        """
        Bar chart of most frequent terms.
        """


        data = frequency_data[:top_n]


        words = [

            item[0]

            for item in data

        ]


        counts = [

            item[1]

            for item in data

        ]



        fig, ax = plt.subplots()



        ax.bar(

            words,

            counts

        )


        ax.set_title(
            "Top Word Frequency"
        )


        ax.set_xlabel(
            "Terms"
        )


        ax.set_ylabel(
            "Frequency"
        )


        plt.xticks(
            rotation=45,
            ha="right"
        )


        return fig



# ------------------------------------------------------
# Evaluation Metrics Chart
# ------------------------------------------------------


    def evaluation_chart(
            self,
            metrics
    ):
        """
        Display IR evaluation metrics.
        """


        names = list(
            metrics.keys()
        )


        values = list(
            metrics.values()
        )



        fig, ax = plt.subplots()



        ax.bar(

            names,

            values

        )


        ax.set_ylim(
            0,
            1
        )


        ax.set_title(
            "IR Evaluation Metrics"
        )


        plt.xticks(
            rotation=45
        )


        return fig



# ------------------------------------------------------
# PageRank Visualization
# ------------------------------------------------------


    def pagerank_chart(self, dataframe):
        """
        PageRank score visualization.
        """

        # Group documents having the same PageRank score
        grouped = (
            dataframe
            .groupby("PageRank Score")["Index"]
            .apply(lambda x: ",".join(map(str, x)))
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.bar(
            grouped["Index"],
            grouped["PageRank Score"]
        )

        ax.set_title("PageRank Importance")
        ax.set_xlabel("Document Index")
        ax.set_ylabel("PageRank Score")

        plt.xticks(rotation=45)

        return fig



# ------------------------------------------------------
# Ranking Comparison Chart
# ------------------------------------------------------


    def ranking_comparison(
            self,
            results
    ):
        """
        Compare:

        TF-IDF score
        vs
        Final ranking score

        """


        df = pd.DataFrame(
            results
        )


        fig, ax = plt.subplots()



        df.plot(

            x="document_id",

            y=[
                "score",
                "final_score"
            ],

            kind="bar",

            ax=ax

        )


        ax.set_title(
            "Ranking Comparison"
        )


        return fig



# ------------------------------------------------------
# Recommendation Similarity Chart
# ------------------------------------------------------


    def recommendation_chart(
            self,
            recommendations
    ):


        df = pd.DataFrame(
            recommendations
        )


        fig, ax = plt.subplots()



        ax.bar(

            df["document_id"],

            df["similarity"]

        )


        ax.set_ylim(
            0,
            1
        )


        ax.set_title(
            "Recommendation Similarity"
        )


        return fig



# ------------------------------------------------------
# Corpus Statistics Table
# ------------------------------------------------------


    def statistics_table(
            self,
            statistics
    ):

        return pd.DataFrame(

            [

                statistics

            ]

        )