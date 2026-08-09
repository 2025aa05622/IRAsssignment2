"""
app.py

End-to-End Information Retrieval System

Streamlit Frontend

Features:

Dashboard
Search
Crawler
Index Management
Ranking
Recommendation
Evaluation
Analytics

"""

import logging
import re

import pandas as pd
import streamlit as st
from modules.crawler import WebCrawler
from modules.database import db
from modules.evaluation import IREvaluator
from modules.indexing import IndexManager
from modules.pagerank import PageRankModel
from modules.preprocessing import TextPreprocessor
from modules.recommender import ContentBasedRecommender
from modules.search_engine import SearchEngine
from modules.visualization import IRVisualizer

# #logging.basicConfig(
#     filename="app.log",
#     filemode="w",
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
#logging.info("Application started")

import os

#st.write("My log file is located at:", os.path.abspath("app.log"))
# ------------------------------------------------------
# Page Configuration
# ------------------------------------------------------

st.set_page_config(
    page_title="Intelligent Information Retrieval System", page_icon="🔎", layout="wide"
)


# ------------------------------------------------------
# Initialize Components
# ------------------------------------------------------


@st.cache_resource
def load_components():

    db.clear_documents()

    return {
        "crawler": WebCrawler(),
        "processor": TextPreprocessor(),
        "index": IndexManager(),
        "pagerank": PageRankModel(),
        "recommender": ContentBasedRecommender(),
        "evaluation": IREvaluator(),
        "visualizer": IRVisualizer(),
    }


components = load_components()


crawler = components["crawler"]

processor = components["processor"]

index = components["index"]

pagerank = components["pagerank"]

recommender = components["recommender"]

evaluation = components["evaluation"]

visualizer = components["visualizer"]


# ------------------------------------------------------
# Session Storage
# ------------------------------------------------------


if "documents" not in st.session_state:

    st.session_state.documents = []


if "search_results" not in st.session_state:

    st.session_state.search_results = []


if "search_results" not in st.session_state:

    st.session_state.retrieved_results = []


# ------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------


st.sidebar.title("IR System Modules")


page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Crawler",
        "Index Management",
        "Search",
        "Ranking",
        "Recommendation",
        "Evaluation",
        "Performance Analytics",
    ],
)

import re

def highlight_text(text: str, pattern: re.Pattern) -> str:
    return pattern.sub(
        lambda m: f"<mark>{m.group(0)}</mark>",
        text,
    )

def create_highlighted_snippet(text, query, snippet_size=250, isPhraseSearch=False):
    """
    Return a snippet around the first occurrence of any query term
    with matching words highlighted.
    """

    query = (
    query.replace("*", "")
         .replace('"', "")
         .replace("'", "")
         .strip()
    )

    start = 0
    end = min(len(text), snippet_size)

    if not text:
        return "No content available."

    if (isPhraseSearch):        
        query_terms = query
    else:        
        query_terms = [t for t in query.split() if t.strip()]        

    if not query_terms:
        return text[:snippet_size]

    if (isPhraseSearch) :
        pattern = re.compile(
            re.escape(query),
            re.IGNORECASE,
        )
    else : 
        pattern = re.compile(
            r"\b(" + "|".join(map(re.escape, query_terms)) + r")\b",
            re.IGNORECASE,
        )

    match = pattern.search(text)

    if match:
        start = max(0, match.start() - snippet_size // 2)
        end = min(len(text), match.end() + snippet_size // 2)
        snippet = text[start:end]
    else:
        snippet = text[:snippet_size]

    highlighted = highlight_text(snippet, pattern)

    if start > 0:
        highlighted = "..." + highlighted

    if end < len(text):
        highlighted += "..."

    return highlighted

# ------------------------------------------------------
# Dashboard
# ------------------------------------------------------


if page == "Dashboard":

    st.title("🔎 Intelligent Information Retrieval System")

    st.write("""
        Complete end-to-end workflow:

        Crawling → Processing → Indexing →
        Searching → Ranking →
        Recommendation → Evaluation
        """)

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric("Documents", len(st.session_state.documents))

    with col2:

        st.metric("Vocabulary", len(index.get_vocabulary()))

    with col3:

        st.metric("Indexed Terms", len(index.inverted_index))

    with col4:

        st.metric("Searches", len(st.session_state.search_results))

    st.subheader("System Pipeline")

    st.code("""
        Web Crawling
              |
              v
        Text Mining
              |
              v
        Inverted Index
              |
              v
        Search Engine
              |
              v
        PageRank Ranking
              |
              v
        Recommendation
              |
              v
        Evaluation
        """)


# ------------------------------------------------------
# Crawler Interface
# ------------------------------------------------------


elif page == "Crawler":

    st.title("🌐 Web Crawling Interface")

    st.write("""
    Acquire documents from heterogeneous web sources.

    Configure

    - Multiple seed URLs
    - Crawling depth
    - Maximum pages
    """)

    urls = st.text_area(
        "Enter URLs (one per line)",
        placeholder="""
        https://example.com
        https://another-site.com
        """,
    )

    crawl_depth = st.slider(
        "Crawling Depth",
        min_value=1,
        max_value=5,
        value=2,
    )

    max_pages = st.number_input(
        "Maximum Pages",
        min_value=1,
        max_value=7,
        value=3,
    )

    if st.button("🚀 Start Crawling"):
        db.clear_documents()
        # Clear Streamlit session
        st.session_state.documents = []

        # Reset crawler state
        crawler.visited_urls = set()
        crawler.total_pages = 0

        if urls.strip() == "":

            st.warning("Please enter at least one URL.")

        else:

            seed_urls = [url.strip() for url in urls.split("\n") if url.strip()]

            with st.spinner("Crawling..."):

                result = crawler.crawl(
                    seed_urls=seed_urls,
                    max_depth=crawl_depth,
                    max_pages=max_pages,
                )

            documents = result.get("Documents", [])

            ####################################################
            # If crawler returned nothing,
            # load everything from DB
            ####################################################

            if len(documents) == 0:

                db_docs = db.get_all_documents()

                documents = []

                for doc in db_docs:

                    documents.append(
                        {
                            "document_id": doc.get("id"),
                            "url": doc.get("url"),
                            "title": doc.get("title"),
                            "content": doc.get("content"),
                        }
                    )

            st.session_state.documents = documents

            st.success(f"""
                Crawling Completed

                Pages Crawled : {result["pages"]}

                Visited URLs : {result["visited"]}

                Documents Available : {len(st.session_state.documents)}
                """)

            st.divider()

    ###########################################################
    # Statistics
    ###########################################################

    if st.session_state.documents:

        st.subheader("Crawl Statistics")

        col1, col2, col3 = st.columns(3)

        unique_urls = len(set(doc["url"] for doc in st.session_state.documents))

        with col1:

            st.metric(
                "Documents",
                len(st.session_state.documents),
            )

        with col2:

            st.metric(
                "Unique URLs",
                unique_urls,
            )

        with col3:

            st.metric(
                "Duplicates Removed",
                len(st.session_state.documents) - unique_urls,
            )

        ###########################################################
        # Metadata Table
        ###########################################################

        metadata = []

        for doc in st.session_state.documents:

            metadata.append(
                {
                    "Document ID": doc.get("document_id"),
                    "URL": doc.get("url"),
                    "Title": doc.get("title", "NA"),
                    "Length": len(doc.get("content", "")),
                }
            )

        st.subheader("Document Metadata")

        st.dataframe(
            pd.DataFrame(metadata),
            use_container_width=True,
        )

        ###########################################################
        # Debug
        ###########################################################

        with st.expander("Returned Documents"):
            st.json(st.session_state.documents)


# ------------------------------------------------------
# Index Management Interface
# ------------------------------------------------------


elif page == "Index Management":

    st.title("📚 Index Management")

    st.write("""
        Manage document processing,
        indexing, and index persistence.
        """)

    if not st.session_state.documents:

        st.warning("""
            No documents available.

            Please crawl documents first.
            """)

    else:

        st.subheader("Document Processing")

        if st.button("⚙️ Build Search Index"):

            with st.spinner("Processing documents..."):

                processed_documents = []

                for doc in st.session_state.documents:

                    cleaned_text = processor.tokens_to_text(
                        processor.preprocess(doc["content"])
                    )

                    processed_documents.append(
                        {
                            "document_id": doc["document_id"],
                            "content": cleaned_text,
                            "url": doc.get("url"),
                            "title": doc.get("title", ""),
                        }
                    )

                st.session_state.processed_documents = processed_documents

                index.build_inverted_index(processed_documents)

                index.calculate_document_frequency()

                index.build_tfidf_matrix(processed_documents)

                index.save_index()

            st.success("Index successfully created")

        st.divider()

        if index.documents:

            st.subheader("Index Statistics")

            statistics = index.index_statistics()

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric("Documents", statistics["documents"])

            with col2:

                st.metric("Unique Terms", statistics["unique_terms"])

            with col3:

                st.metric("Postings", statistics["total_postings"])

            with col4:

                st.metric("Avg Doc Length", statistics["average_document_length"])

            st.subheader("Vocabulary Sample")

            vocabulary = index.get_vocabulary()

            st.write(vocabulary[:100])

            st.subheader("Load Existing Index")

            if st.button("📥 Load Index"):

                try:

                    index.load_index()

                    st.success("Index loaded successfully")

                except Exception as e:

                    st.error(str(e))


# ------------------------------------------------------
# Search Interface
# ------------------------------------------------------


elif page == "Search":

    st.title("🔎 Intelligent Search Interface")

    st.write("""
        Search indexed documents using
        multiple retrieval strategies.
        """)

    if not index.documents:

        st.warning("""
            Search index not available.

            Please build index first.
            """)

    else:

        query = st.text_input(
            "Enter your search query", placeholder="Example: artificial intelligence"
        )

        col1, col2 = st.columns(2)

        with col1:

            search_mode = st.selectbox(
                "Search Method", ["ranked", "keyword", "phrase", "wildcard", "fuzzy"]
            )

        with col2:

            top_k = st.number_input(
                "Number of Results", min_value=1, max_value=50, value=10
            )

        if st.button("🚀 Search"):

            if query.strip() == "":

                st.warning("Enter a search query")

            else:

                with st.spinner("Searching..."):

                    search_engine = SearchEngine(index, processor, st.session_state.documents)

                    results, retrieved_results = search_engine.search(query, mode=search_mode, top_k=top_k)

                    st.session_state.search_results = results
                    st.session_state.retrieved_results = retrieved_results

                st.success(f"{len(results)} results found")

        st.divider()

        isPhraseSearch = False

        if search_mode == "phrase":
            isPhraseSearch = True
        else:
            isPhraseSearch = False

        if st.session_state.search_results:

            st.subheader("Search Results")

            # Create a lookup dictionary: document_id
            doc_lookup = {
                doc["document_id"]: doc
                for doc in st.session_state.documents
            }

            for i, result in enumerate(st.session_state.search_results):

                with st.container():

                    doc_id = result.get("document_id")
                    document = doc_lookup.get(doc_id, {})
                    url = document.get("url", "URL not found")
                    content = document.get("content", "")

                    snippet = create_highlighted_snippet(content, query, 250, isPhraseSearch)
                    

                    st.markdown(f"""
                    ### Result {i+1}

                    **Document ID:** {doc_id}

                    **URL:** {url}
                    """)

                    st.markdown("**Content Preview:**", unsafe_allow_html=True)
                    st.markdown(snippet, unsafe_allow_html=True)

                    if "score" in result:
                        st.metric("Relevance Score", result["score"])

                    st.divider()


# ------------------------------------------------------
# Ranking Visualization
# ------------------------------------------------------


elif page == "Ranking":

    st.title("📈 Ranking Visualization")

    st.write("""
        Analyze document importance using:

        1. Content relevance (TF-IDF)
        2. Page importance (PageRank)

        Combined ranking improves search quality.
        """)

    if not st.session_state.documents:

        st.warning("No documents available. Crawl documents first.")

    else:

        st.subheader("Build Web Graph")

        st.info("""
            PageRank requires link information.

            Crawled documents with outgoing links
            are used to build the graph.
            """)

        if st.button("Calculate PageRank"):

            pages = []

            for doc in st.session_state.documents:

                pages.append(
                    {
                        "url": doc.get("url", str(doc["document_id"])),
                        "links": doc.get("links", []),
                    }
                )

            pagerank.build_graph(pages)

            scores = pagerank.calculate_pagerank()

            st.session_state.pagerank_scores = scores

            st.success("PageRank calculated successfully")

        st.divider()

        if hasattr(st.session_state, "pagerank_scores"):

            st.subheader("PageRank Scores")

            ranking_df = pagerank.ranking_dataframe()

            st.dataframe(ranking_df, use_container_width=True)

            st.subheader("Authority Visualization")

            fig = visualizer.pagerank_chart(ranking_df)

            st.pyplot(fig)

        st.divider()

        st.subheader("Ranking Comparison")

        st.write("""
            Example:

            Document A:
            High TF-IDF but low authority


            Document B:
            Slightly lower TF-IDF but
            high PageRank


            Combined ranking can move B higher.
            """)


# ------------------------------------------------------
# Recommendation Panel
# ------------------------------------------------------


elif page == "Recommendation":

    st.title("🎯 Recommendation System")

    st.write("""
        Content-based recommendation using:

        TF-IDF document representation

        +

        Cosine similarity
        """)

    if not index.documents:

        st.warning("""
            No indexed documents available.

            Please build index first.

            """)

    else:

        documents = list(index.documents.keys())

        selected_document = st.selectbox("Select Document", documents)

        top_k = st.number_input(
            "Number of Recommendations", min_value=1, max_value=4, value=4
        )

        if st.button("Generate Recommendations"):

            document_list = []

            for doc_id, content in index.documents.items():

                document_list.append({"document_id": doc_id, "content": content})

            recommender.fit(document_list)

            recommendations = recommender.recommend(selected_document, top_k)

            st.session_state.recommendations = recommendations

            st.success("Recommendations generated")

        st.divider()

        if "recommendations" in st.session_state:

            st.subheader("Top-K Recommendations")

            recommendation_df = pd.DataFrame(st.session_state.recommendations)

            st.dataframe(recommendation_df, use_container_width=True)

            st.subheader("Similarity Visualization")

            fig = visualizer.recommendation_chart(st.session_state.recommendations)

            st.pyplot(fig)

            stats = recommender.recommendation_statistics(
                st.session_state.recommendations
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric("Recommendations", stats["recommendations"])

            with col2:

                st.metric("Average Similarity", stats["average_similarity"])

            with col3:

                st.metric("Highest Similarity", stats["highest_similarity"])


# ------------------------------------------------------
# Evaluation Dashboard
# ------------------------------------------------------


elif page == "Evaluation":

    st.title("📊 Information Retrieval Evaluation")

    st.write("""
        Evaluate retrieval effectiveness
        using standard IR metrics.
        """)

    st.info("""
        For evaluation, provide:

        1. Retrieved document IDs
        2. Relevant document IDs

        The system compares retrieved
        results against ground truth.
        """)

    if (
        not index.documents
        or not st.session_state.search_results
        or not st.session_state.retrieved_results
    ):

        st.warning("No indexed documents available.")

    else:

        st.subheader("Evaluation Input")

        relevant_docs = st.session_state.search_results
        retrieved_docs = st.session_state.retrieved_results

        retrieved_ids = ",".join(str(doc["ret_document_id"]) for doc in retrieved_docs)
        relevant_ids = ",".join(str(doc["document_id"]) for doc in relevant_docs)

        retrieved_input = st.text_input(
            "Retrieved Documents",
            value=retrieved_ids,
            disabled=True,
        )

        relevant_input = st.text_input(
            "Relevant Documents",
            value=relevant_ids,
            disabled=True,
        )

        k_value = st.slider("K Value", min_value=1, max_value=5, value=5)

        calculatemetrics = st.button(
            "Calculate Metrics",
            disabled=(len(relevant_docs) == 0 or len(retrieved_docs) == 0)
        )


        if calculatemetrics:

            try:

                retrieved = [int(x.strip()) for x in retrieved_input.split(",")]

                relevant = [int(x.strip()) for x in relevant_input.split(",")]

                report = evaluation.evaluation_report(retrieved, relevant, k_value)

                st.session_state.evaluation_result = report

                st.success("Evaluation completed")

            except Exception as e:

                st.error(str(e))

        st.divider()

        if "evaluation_result" in st.session_state:

            st.subheader("Evaluation Results")

            metrics_df = pd.DataFrame([st.session_state.evaluation_result])

            st.dataframe(metrics_df, use_container_width=True)

            st.subheader("Metric Visualization")

            fig = visualizer.evaluation_chart(st.session_state.evaluation_result)

            st.pyplot(fig)

        st.divider()

        st.subheader("Ranking Quality Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric("MAP", "Calculated during batch evaluation")

        with col2:

            st.metric("MRR", "First relevant result")

        with col3:

            st.metric("NDCG", "Ranking quality")


# ------------------------------------------------------
# Performance Analytics Dashboard
# ------------------------------------------------------


elif page == "Performance Analytics":

    st.title("⚡ Performance Analytics")

    st.write("""
        Monitor system performance,
        scalability, and efficiency.
        """)

    # ------------------------------------------
    # Document Analytics
    # ------------------------------------------

    st.subheader("📄 Document Analytics")

    if st.session_state.documents:

        total_documents = len(st.session_state.documents)

        total_words = sum(
            len(doc.get("content", "").split()) for doc in st.session_state.documents
        )

        avg_length = total_words / total_documents

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric("Total Documents", total_documents)

        with col2:

            st.metric("Total Words", total_words)

        with col3:

            st.metric("Average Document Length", round(avg_length, 2))

    else:

        st.info("No documents crawled yet.")

    st.divider()

    # ------------------------------------------
    # Index Analytics
    # ------------------------------------------

    st.subheader("📚 Index Analytics")

    if index.documents:

        stats = index.index_statistics()

        index_df = pd.DataFrame([stats])

        st.dataframe(index_df, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:

            st.metric("Vocabulary Size", stats["unique_terms"])

        with col2:

            st.metric("Posting Entries", stats["total_postings"])

    else:

        st.info("Index not created.")

    st.divider()

    # ------------------------------------------
    # Search Analytics
    # ------------------------------------------

    st.subheader("🔎 Search Analytics")

    if st.session_state.search_results:

        search_stats = SearchEngine(index, processor, st.session_state.documents).search_statistics(
            st.session_state.search_results
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric("Results Returned", search_stats["results"])

        with col2:

            st.metric("Average Score", search_stats["average_score"])

        with col3:

            st.metric("Highest Score", search_stats["highest_score"])

    else:

        st.info("No searches performed yet.")

    st.divider()

    # ------------------------------------------
    # System Pipeline Health
    # ------------------------------------------

    st.subheader("🚦 System Component Status")

    components_status = {
        "Crawler": "Available",
        "Preprocessor": "Available",
        "Indexer": "Available" if index.documents else "Waiting",
        "Search Engine": "Available",
        "Ranking": "Available",
        "Recommendation": "Available",
        "Evaluation": "Available",
    }

    status_df = pd.DataFrame(
        list(components_status.items()), columns=["Component", "Status"]
    )

    st.table(status_df)
