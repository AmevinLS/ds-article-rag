import streamlit as st
import pandas as pd

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.docstore.document import Document
import ollama

from retrieval import faiss_from_docs

from typing import List, Tuple
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
class RetrieveResult:
    docs_with_scores: List[Tuple[Document, float]]
    sim_threshold: float
    retrieve_k: int

    def filter_docs_by_score(self) -> List[Tuple[Document, float]]:
        result = [
            (doc, score)
            for doc, score in self.docs_with_scores
            if score >= self.sim_threshold
        ]
        return result


class Response:
    def __init__(
        self,
        query: str,
        retr_result: RetrieveResult,
        articles_df: pd.DataFrame,
        query_llm: bool = False,
    ):
        self.segment_entries = []
        self.n_not_shown = 0
        for i, (doc, score) in enumerate(retr_result.docs_with_scores):
            if score < retr_result.sim_threshold:
                self.n_not_shown += 1
            else:
                self.segment_entries.append(
                    {
                        "score": score,
                        "title": self._get_article_title(doc, articles_df),
                        "content": doc.page_content[6:],
                    }
                )
        self.llm_response = None
        if query_llm and len(self.segment_entries) > 0:
            with st.spinner("Querying LLM..."):
                self.llm_response = self._prompt_ollama(query, "llama2")

    def _prompt_ollama(self, query: str, model: str) -> str:
        prompt_system = "Summarize the information to the user's prompt based only on the retrieved documents below. Do not add any external information.\n\n"
        for i, entry in enumerate(self.segment_entries):
            prompt_system += f"{i+1}. {entry['title']}\n\n" f"{entry['content']}\n\n"

        ollama_result = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": prompt_system},
                {
                    "role": "user",
                    "content": query,
                },
            ],
        )
        return ollama_result["message"]["content"]

    def _get_article_title(self, doc: Document, articles_df: pd.DataFrame) -> str:
        return articles_df.loc[int(doc.metadata["article_idx"]), "Title"]

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
            message["content"].display()

    ## Process user prompt
    if prompt := st.chat_input("Please write something"):
        st.chat_message("user").markdown(prompt)
        messages.append({"role": "user", "content": prompt})

        docs_with_scores = faiss_db.similarity_search_with_score(prompt, k=retrieve_k)
        response = Response(
            query=prompt,
            retr_result=RetrieveResult(docs_with_scores, sim_threshold, retrieve_k),
            articles_df=art_df,
            query_llm=query_llm,
        )
        response.display()
        messages.append({"role": "assisstant", "content": response})
