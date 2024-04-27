import streamlit as st

from langchain.document_loaders.csv_loader import CSVLoader

from retrieval import faiss_from_docs


def build_faiss(csv_path: str):
    loader = CSVLoader(
        csv_path, 
        metadata_columns=["Title"], 
        encoding="utf-8"
    )
    articles = loader.load()

    return faiss_from_docs(
        article_docs=articles,
        model_name="all-MiniLM-L6-v2"
    )


if __name__ == "__main__":
    st.title("Towards Data Science Articles Retrieval")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "faiss_db" not in st.session_state:
        st.session_state.faiss_db = build_faiss(R"D:\Repos\ds-article-rag\data\medium.csv")

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])


    if prompt := st.chat_input("Please write something"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        docs = st.session_state.faiss_db.similarity_search(prompt, k=4)
        response = "Relevant articles:"
        for i, doc in enumerate(docs):
            response += f"\n {i+1}. {doc.metadata['Title']}"
        
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assisstant", "content": response})