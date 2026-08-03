"""
pagerank.py

PageRank based ranking module.

Used to calculate importance
of web pages/documents based on
link structure.

"""

import networkx as nx

import pandas as pd



class PageRankModel:


    def __init__(
            self,
            damping_factor=0.85,
            iterations=100
    ):


        self.damping_factor = damping_factor

        self.iterations = iterations

        self.graph = nx.DiGraph()

        self.scores = {}



# ------------------------------------------------------
# Build Web Graph
# ------------------------------------------------------


    def build_graph(
            self,
            documents
    ):
        """
        Create directed graph.

        documents format:


        [
          {
            "url":
            "A.com",

            "links":
            [
              "B.com",
              "C.com"
            ]
          }
        ]

        """


        for doc in documents:


            source = doc["url"]


            self.graph.add_node(
                source
            )



            for link in doc.get(
                    "links",
                    []
            ):


                self.graph.add_edge(

                    source,

                    link

                )



        return self.graph



# ------------------------------------------------------
# Calculate PageRank
# ------------------------------------------------------


    def calculate_pagerank(self):

        """
        Apply PageRank algorithm.

        Formula:

        PR(A)=

        (1-d)/N

        +

        d * Σ(PR(T)/C(T))


        d = damping factor

        """


        self.scores = nx.pagerank(

            self.graph,

            alpha=self.damping_factor,

            max_iter=self.iterations

        )


        return self.scores



# ------------------------------------------------------
# Rank Pages
# ------------------------------------------------------


    def rank_pages(
            self,
            top_k=10
    ):

        """
        Return highest ranked pages.
        """


        ranked = sorted(

            self.scores.items(),

            key=lambda x:x[1],

            reverse=True

        )


        return ranked[:top_k]



# ------------------------------------------------------
# Convert Ranking To DataFrame
# ------------------------------------------------------


    def ranking_dataframe(self):

        data = []

        for idx, (url, score) in enumerate(self.scores.items(), start=1):

            data.append(
                {
                    "Index": idx,
                    "URL": url,
                    "PageRank Score": round(score, 6)
                }
            )

        return pd.DataFrame(data)



# ------------------------------------------------------
# Combine With Search Score
# ------------------------------------------------------


    def combine_ranking(
            self,
            search_results,
            weight=0.3
    ):
        """
        Combine:

        TF-IDF relevance score

        +

        PageRank importance


        Final Score:

        0.7*relevance
        +
        0.3*importance

        """


        combined = []



        for result in search_results:


            url = result.get(
                "url"
            )


            pagerank_score = (

                self.scores
                .get(
                    url,
                    0
                )

            )



            final_score = (

                (1-weight)
                *
                result["score"]

                +

                weight
                *
                pagerank_score

            )



            result["pagerank"] = (

                round(
                    pagerank_score,
                    5
                )

            )


            result["final_score"] = (

                round(
                    final_score,
                    5
                )

            )


            combined.append(
                result
            )



        combined.sort(

            key=lambda x:
            x["final_score"],

            reverse=True

        )


        return combined