from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders.dataframe import DataFrameLoader
from langchain_community.docstore.document import Document
from langchain.vectorstores.utils import DistanceStrategy

import pandas as pd

from typing import List


def faiss_from_docs(article_docs: List[Document], model_name: str) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.from_documents(
        article_docs, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
    )
    return db


def faiss_from_csv(
    csv_path: str, metadata_columns: List[str], model_name: str = "all-MiniLM-L6-v2"
) -> FAISS:
    loader = CSVLoader(csv_path, metadata_columns=metadata_columns, encoding="utf-8")
    articles = loader.load()

    return faiss_from_docs(article_docs=articles, model_name=model_name)


def faiss_from_df(df: pd.DataFrame, page_content_column: str, model_name: str) -> FAISS:
    loader = DataFrameLoader(df, page_content_column=page_content_column)
    articles = loader.load()

    return faiss_from_docs(article_docs=articles, model_name=model_name)


# def faiss_from_cache(data_dir: str):
