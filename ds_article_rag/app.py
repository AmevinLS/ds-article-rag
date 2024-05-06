import streamlit as st

from langchain.docstore.document import Document

import os
from typing import List, Tuple
from dataclasses import dataclass

from data_prep import preprocess
from retrieval import faiss_from_df
from generation import prompt_ollama_with_articles


DATA_DIR = os.environ.get("DATA_DIR", os.path.join(__file__, "..", "data"))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


@dataclass
class RetrieveResult:
    docs_with_scores: List[Tuple[Document, float]]
    sim_threshold: float
    retrieve_k: int


class Response:
    def __init__(
        self,
        query: str,
        retr_result: RetrieveResult,
        query_llm: bool = False,
    ):
        self.segment_entries = []
        self.n_not_shown = 0
        for doc, score in retr_result.docs_with_scores:
            if score < retr_result.sim_threshold:
                self.n_not_shown += 1
            else:
                self.segment_entries.append(
                    {
                        "score": score,
                        "title": doc.metadata["Title"],
                        "content": doc.page_content,
                    }
                )
        self.llm_response = None
        if query_llm and len(self.segment_entries) > 0:
            with st.spinner("Querying LLM..."):
                self.llm_response = prompt_ollama_with_articles(
                    query,
                    model="llama2",
                    articles=self.segment_entries,
                    ollama_host=OLLAMA_HOST,
                )

    def display(self):
        with st.chat_message("assistant"):
            st.markdown("Relevant article segments:")
            for i, entry in enumerate(self.segment_entries):
                st.markdown(
                    f"#### {i+1}. {entry['title']}\n"
                    f"*Score: {entry['score']:.4f}*\n\n"
                )
                c = st.container(border=True)
                c.markdown(f"{entry['content']}\n\n")
            if self.n_not_shown > 0:
                st.markdown(
                    f"*({self.n_not_shown} results were not shown "
                    "because their score was lower than set threshold)*\n\n"
                )
            if self.llm_response is not None:
                st.markdown(f"#### LLM RESPONSE:\n" f"{self.llm_response}")


faiss_from_df = st.cache_resource(faiss_from_df)
preprocess = st.cache_resource(preprocess)

if __name__ == "__main__":
    st.title("Towards Data Science Articles Retrieval")

    ## Sidebar
    st.sidebar.header("Controls")
    st.sidebar.markdown("## **Retrieval Parameters**")
    retrieve_k = st.sidebar.slider(
        "*Set number of article segments to retrieve*",
        value=4,
        min_value=1,
        max_value=10,
        step=1,
        key="retrieve_k",
    )
    sim_threshold = st.sidebar.slider(
        "*Set minimum cosine similarity score threshold*",
        value=0.5,
        min_value=-1.0,
        max_value=1.0,
        key="sim_threshold",
    )

    st.sidebar.markdown("## **Preprocessing Parameters**")
    join_std_multiplier = st.sidebar.slider(
        "*Paragraph joining STD multiplier*",
        value=0.1,
        min_value=-3.0,
        max_value=3.0,
        step=0.1,
        key="join_std_multiplier",
        help="Higher value -> less paragraphs are joined",
    )
    rerun_preprocessing = st.sidebar.button("Rerun Preprocessing", type="primary")
    if ("join_std_mult" not in st.session_state) or rerun_preprocessing:
        st.session_state.join_std_mult = join_std_multiplier

    st.sidebar.markdown("-----")
    query_llm = st.sidebar.toggle(
        "Query LLM on retrieval results\n", help="Will lead to long response times!"
    )
    if st.sidebar.button("Clear history"):
        st.session_state.messages = []

    # Initialize message history if doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
    messages: list = st.session_state.messages

    # TODO: perhaps build and persist faiss beforehand - and only load here
    with st.spinner("Preprocessing data..."):
        joined_pars_df = preprocess(
            DATA_DIR,
            join_std_multiplier=st.session_state.join_std_mult,
            verbose=False,
        )
    with st.spinner("Building FAISS index..."):
        faiss_db = faiss_from_df(
            joined_pars_df, page_content_column="Text", model_name="all-MiniLM-L6-v2"
        )

    ## Show previous messages
    for message in messages:
        if message["role"] == "user":
            st.chat_message(message["role"]).markdown(message["content"])
        else:
            message["content"].display()

    ## Process user prompt
    if prompt := st.chat_input("Please write something"):
        st.chat_message("user").markdown(prompt)
        messages.append({"role": "user", "content": prompt})

        with st.spinner("Retrieving documents from vector store..."):
            docs_with_scores = faiss_db.similarity_search_with_score(
                prompt, k=retrieve_k
            )
        response = Response(
            query=prompt,
            retr_result=RetrieveResult(docs_with_scores, sim_threshold, retrieve_k),
            query_llm=query_llm,
        )
        response.display()
        messages.append({"role": "assisstant", "content": response})
