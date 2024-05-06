from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders.dataframe import DataFrameLoader
from langchain_community.docstore.document import Document
from langchain.vectorstores.utils import DistanceStrategy

import pandas as pd

from typing import List


def faiss_from_docs(article_docs: List[Document], model_name: str) -> FAISS:
    """
    Creates a FAISS index from a list of Langchain Document objects.

    Args:
        article_docs (List[Document]): A list of Langchain Document objects containing the text data and any relevant metadata.
        model_name (str): The name of the pre-trained Hugging Face model to use for generating embeddings.

    Returns:
        FAISS: A FAISS index instance for efficient document retrieval.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.from_documents(
        article_docs, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
    )
    return db


def faiss_from_csv(
    csv_path: str, metadata_columns: List[str], model_name: str
) -> FAISS:
    """
    Creates a FAISS index from a CSV file containing document data.

    Args:
        csv_path (str): The path to the CSV file containing the documents.
        metadata_columns (List[str]): A list of column names in the CSV file that represent document metadata.
        model_name (str): The name of the pre-trained Hugging Face model to use for generating embeddings.

    Returns:
        FAISS: A FAISS index instance for efficient document retrieval.
    """
    loader = CSVLoader(csv_path, metadata_columns=metadata_columns, encoding="utf-8")
    articles = loader.load()

    return faiss_from_docs(article_docs=articles, model_name=model_name)


def faiss_from_df(df: pd.DataFrame, page_content_column: str, model_name: str) -> FAISS:
    """
    Creates a FAISS index from a pandas DataFrame containing document data.

    Args:
        df (pd.DataFrame): A pandas DataFrame holding the document data.
        page_content_column (str): The name of the column in the DataFrame that contains the text content of each document (e.g., "content", "body").
        model_name (str): The name of the pre-trained Hugging Face model to use for generating embeddings.

    Returns:
        FAISS: A FAISS index instance for efficient document retrieval.
    """
    loader = DataFrameLoader(df, page_content_column=page_content_column)
    articles = loader.load()

    return faiss_from_docs(article_docs=articles, model_name=model_name)


# def faiss_from_cache(data_dir: str):
