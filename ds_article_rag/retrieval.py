from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS

from langchain_community.docstore.document import Document
from langchain.vectorstores.utils import DistanceStrategy
from typing import List


def faiss_from_docs(article_docs: List[Document], model_name: str) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.from_documents(
        article_docs, 
        embeddings, 
        distance_strategy=DistanceStrategy.COSINE
    )
    return db
