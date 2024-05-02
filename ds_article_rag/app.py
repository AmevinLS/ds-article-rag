import streamlit as st
import pandas as pd

from langchain.document_loaders.csv_loader import CSVLoader

from retrieval import faiss_from_docs

from typing import List


@st.cache_resource
def build_faiss(csv_path: str, metadata_columns: List[str]):
    loader = CSVLoader(csv_path, metadata_columns=metadata_columns, encoding="utf-8")
    articles = loader.load()

    return faiss_from_docs(article_docs=articles, model_name="all-MiniLM-L6-v2")


@st.cache_data
def get_orig_articles(csv_path: str) -> pd.DataFrame:
    art_df = pd.read_csv(csv_path, encoding="utf-8")
    return art_df


if __name__ == "__main__":
    st.title("Towards Data Science Articles Retrieval")

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
        st.chat_message(message["role"]).markdown(message["content"])

    ## Process user prompt
    if prompt := st.chat_input("Please write something"):
        st.chat_message("user").markdown(prompt)
        messages.append({"role": "user", "content": prompt})

        docs = faiss_db.similarity_search(prompt, k=4)
        response = "Relevant article segments:\n"
        for i, doc in enumerate(docs):
            response += (
                f"#### {i+1}. {art_df.loc[int(doc.metadata['article_idx']), 'Title']}\n"
            )
            response += f"{doc.page_content[6:]}\n\n"
            response += "-----\n\n"  # + "=" * 80

        st.chat_message("assistant").markdown(response)
        messages.append({"role": "assisstant", "content": response})
