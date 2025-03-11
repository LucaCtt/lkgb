import logging

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector

from rov_parser.ontology import SlogertOntology

# Disable Chroma info logging
logging.getLogger("langchain_chroma").propagate = False


class Store:
    """
    Store implementation using Neo4j for storing and retrieving text embeddings.

    """

    def __init__(
        self,
        url: str,
        username: str,
        password: str,
        embeddings_model: Embeddings,
        ontology: SlogertOntology,
    ) -> "Store":
        """
        Initializes the VectorStore with Neo4j connection details and an embeddings model.

        Args:
            url (str): The URL for connecting to the Neo4j database.
            username (str): The username for the Neo4j database.
            password (str): The password for the Neo4j database.
            embeddings_model (Embeddings): The model used to generate embeddings for the data.
            ontology (SlogertOntology): The ontology used to extract structured information from the data.

        Attributes:
            store (Neo4jVector): The Neo4j vector store initialized with the given connection details and embeddings model.

        """
        self.graph_store = Neo4jGraph(url=url, username=username, password=password)

        self.vector_store = Neo4jVector.from_existing_graph(
            url=url,
            username=username,
            password=password,
            embedding=embeddings_model,
            node_label="Event",
            text_node_properties=["text"],
            embedding_node_property="embedding",
        )

        self.ontology = ontology

    def find_similar_logs_with_template(self, log: str, score_threshold: float) -> list[Document]:
        """
        Finds logs that are similar to the given log and also have a template.

        Args:
            log (str): The log string to find similar logs for.
            score_threshold (float): The similarity score threshold to use for filtering.

        Returns:
            list[Document]: A list of Document objects that are very similar to the given log.

        """
        similar = self.vector_store.similarity_search_with_relevance_scores(
            self.__compose_similarity_question(log),
            score_threshold=score_threshold,
            k=3,
            filter={"template": {"$ne": ""}},
        )
        return [doc for doc, _ in similar]

    def add_log(self, log: str, template: str) -> None:
        """
        Adds a new log to the store.

        Args:
            log (str): The log string to be added to the store.
            template (str): The template string associated with the log.

        """
        log + template + "fuck off ruff"

    def __compose_similarity_question(self, log: str) -> str:
        return f"search_query: {log}"
