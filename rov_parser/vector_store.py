import logging
import uuid

import pandas as pd
from chromadb import Embeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.documents import Document

# Disable Chroma info logging
logging.getLogger("langchain_chroma").propagate = False


class VectorStore:
    """
    VectorStore is thin wrapper around Chroma for storing and retrieving text embeddings.
    """

    def __init__(self, persist_dir: str | None, embeddings_model: Embeddings) -> "VectorStore":
        """
        Initializes the VectorStore with a persistent directory and an embeddings model.

        Args:
            persist_dir (str): The directory where the vector store will persist its data.
            embeddings_model (Embeddings): The model used to generate embeddings for the data.

        Attributes:
            store (Chroma): The Chroma vector store initialized with the given directory and embeddings model.
            splitter (RecursiveCharacterTextSplitter): A text splitter that splits text into chunks of size 1000 with an overlap of 100 characters.

        """
        self.store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings_model,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def load_df(self, df: pd.DataFrame) -> None:
        """
        Loads a DataFrame into the vector store.

        This method takes a pandas DataFrame, uses a DataFrameLoader to extract
        the content from the specified column, and then adds the extracted data
        to the vector store.

        Args:
            df (pd.DataFrame): The DataFrame containing the data to be loaded.
                               The DataFrame should have a column named "text"
                               which contains the content to be processed.

        Returns:
            None

        """
        loader = DataFrameLoader(data_frame=df, page_content_column="text")
        data = loader.load()

        self.store.add_documents(data)

    def get_template(self, log: str) -> str | None:
        """
        Finds an exact match for the given log in the store.

        Args:
            log (str): The log string to find an exact match for.

        Returns:
            Document: A Document object that is an exact match for the given log.

        """
        match = self.store.get(where_document={"$contains": log})
        if not match["documents"]:
            return None

        return match["metadatas"][0]["template"]

    def find_very_similar_logs_with_template(self, log: str) -> list[Document]:
        """
        Finds logs that are very similar to the given log using a predefined template.

        This method searches for logs in the store that have a high similarity score
        with the provided log. It uses a similarity threshold and filters out logs
        that do not have a template.

        Args:
            log (str): The log string to find similar logs for.

        Returns:
            list[Document]: A list of Document objects that are very similar to the given log.

        """
        similar = self.store.similarity_search_with_relevance_scores(
            self.__compose_similarity_question(log),
            score_threshold=0.7,
            k=10,
            filter={"template": {"$ne": ""}},
        )
        return [doc for doc, _ in similar]

    def find_similar_logs(self, log: str) -> list[Document]:
        """
        Find logs similar to the given log.

        This method searches for logs that are similar to the provided log string
        using a similarity search algorithm. It returns a list of Document objects
        that are considered similar based on a relevance score threshold.

        Args:
            log (str): The log string to find similar logs for.

        Returns:
            list[Document]: A list of Document objects that are similar to the given log.

        """
        similar = self.store.similarity_search_with_relevance_scores(
            self.__compose_similarity_question(log),
            score_threshold=0.4,
            k=5,
        )
        return [doc for doc, _ in similar]

    def add_document(self, log: str, template: str) -> None:
        """
        Adds a new document to the store.

        Args:
            log (str): The log string to be added to the store.
            template (str): The template string associated with the log.

        """
        document = (
            Document(
                id=uuid.uuid4(),
                page_content="search_document: " + log,
                metadata={"template": template},
            ),
        )

        self.store.add_documents([document])

    def update_document(self, document: Document) -> None:
        """
        Updates an existing document in the store.

        Args:
            document (Document): The document object containing updated information.

        """
        self.store.update_document(document_id=document.id, document=document)

    def get_documents_without_template(self) -> list[Document]:
        """
        Retrieves logs from the store that do not have a template.

        Returns:
            list[Document]: A list of Document objects that do not have a template.

        """
        return self.store.get(where={"template": {"$eq": ""}})["documents"]

    def __compose_similarity_question(self, log: str) -> str:
        return f"search_query: {log}"
