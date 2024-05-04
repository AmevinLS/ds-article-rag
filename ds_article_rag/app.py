import streamlit as st
import pandas as pd

from langchain.document_loaders.csv_loader import CSVLoader

from retrieval import faiss_from_docs

from typing import List, Tuple, Any
from dataclasses import dataclass


@st.cache_resource
def build_faiss(csv_path: str, metadata_columns: List[str]):
    loader = CSVLoader(csv_path, metadata_columns=metadata_columns, encoding="utf-8")
    articles = loader.load()

    return faiss_from_docs(article_docs=articles, model_name="all-MiniLM-L6-v2")


@st.cache_data
def get_orig_articles(csv_path: str) -> pd.DataFrame:
    art_df = pd.read_csv(csv_path, encoding="utf-8")
    return art_df


@dataclass
class Response:
    docs_with_scores: List[Tuple[Any, float]]
    sim_threshold: float


def display_results(response: Response):
    with st.chat_message("assistant"):
        st.markdown("Relevant article segments:")
        for i, (doc, score) in enumerate(response.docs_with_scores):
            if score < response.sim_threshold:
                st.markdown(
                    f"*({len(response.docs_with_scores) - i} results were not shown "
                    "because their score was lower than set threshold)*\n\n"
                )
                break
            st.markdown(
                f"#### {i+1}. {art_df.loc[int(doc.metadata['article_idx']), 'Title']}\n"
                f"*Score: {score:.4f}*\n\n"
            )
            c = st.container(border=True)
            c.markdown(f"{doc.page_content[6:]}\n\n")
            # c.markdown(f"```text\n{doc.page_content[6:]}\n```\n")
            # st.markdown(f"> ```{doc.page_content[6:]}```")


if __name__ == "__main__":
    st.title("Towards Data Science Articles Retrieval")

    ## Sidebar
    st.sidebar.header("Controls")
    retrieve_k = st.sidebar.slider(
        "Set number of article segments to retrieve",
        value=4,
        min_value=1,
        max_value=10,
        step=1,
        key="retrieve_k",
    )
    sim_threshold = st.sidebar.slider(
        "Set minimum cosine similarity threshold",
        value=0.5,
        min_value=-1.0,
        max_value=1.0,
        key="sim_threshold",
    )
    if st.sidebar.button("Clear history"):
        st.session_state.messages = []

    # Initialize message history if doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
    messages: list = st.session_state.messages

    # TODO: perhaps build and persist faiss beforehand - and only load here
    faiss_db = build_faiss(
        R"D:\Repos\ds-article-rag\data\grouped_paragraphs.csv",
        metadata_columns=["article_idx"],
    )
    art_df = get_orig_articles(R"D:\Repos\ds-article-rag\data\medium.csv")

    ## Show previous messages
    for message in messages:
        if message["role"] == "user":
            st.chat_message(message["role"]).markdown(message["content"])
        else:
            display_results(message["content"])

    ## Process user prompt
    if prompt := st.chat_input("Please write something"):
        st.chat_message("user").markdown(prompt)
        messages.append({"role": "user", "content": prompt})

        docs_with_scores = faiss_db.similarity_search_with_score(prompt, k=retrieve_k)
        response = Response(docs_with_scores, sim_threshold)
        display_results(response)
        messages.append({"role": "assisstant", "content": response})

    # st.chat_message("system").markdown(f"```\n#This is a comment\nhabibib")
