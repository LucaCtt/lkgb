import re
import uuid

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship

from rov_parser.ontology import Ontology


class Store:
    """Store implementation wrapping both the Neo4j graph database and vector index.

    Allows for storing and retrieving text embeddings and graph data.
    """

    def __init__(
        self,
        url: str,
        username: str,
        password: str,
        embeddings_model: Embeddings,
        ontology: Ontology,
    ) -> "Store":
        """Initialize the VectorStore with Neo4j connection details and an embeddings model.

        Args:
            url (str): The URL for connecting to the Neo4j database.
            username (str): The username for the Neo4j database.
            password (str): The password for the Neo4j database.
            embeddings_model (Embeddings): The model used to generate embeddings for the data.
            ontology (Ontology): The ontology used to infer graph data from log events.

        Attributes:
            store (Neo4jVector): The Neo4j vector store initialized with the given
            connection details and embeddings model.

        """
        self.ontology = ontology

        self.graph_store = Neo4jGraph(url=url, username=username, password=password)

        self.vector_store = Neo4jVector.from_existing_graph(
            url=url,
            username=username,
            password=password,
            embedding=embeddings_model,
            node_label=self.ontology.event_class_name,
            text_node_properties=["value"],
            embedding_node_property="embedding",
        )

    def clear(self) -> None:
        """Clear the store of all data."""
        self.graph_store.query("MATCH (n) DETACH DELETE n")

    def get_template(self, event: str) -> str | None:
        """Get the template associated with a log event.

        Args:
            event (str): The log event string to get the template for.

        Returns:
            str: The template associated with the log, or None if no template is found.

        """
        res = self.graph_store.query(
            f"""
            MATCH (l:{self.ontology.event_class_name} {{value: $value}})
            RETURN l.template AS template
            """,
            params={"value": event},
        )

        return res.single()["template"] if res.single() else None

    def find_similar_events_with_template(self, event: str, score_threshold: float) -> list[Document]:
        """Find logs that are similar to the given log and also have a template.

        Args:
            event (str): The log event string to find similar logs for.
            score_threshold (float): The similarity score threshold to use for filtering.

        Returns:
            list[Document]: A list of Document objects that are very similar to the given log.

        """
        similar = self.vector_store.similarity_search_with_relevance_scores(
            self.__compose_similarity_question(event),
            score_threshold=score_threshold,
            k=3,
            filter={"template": {"$ne": ""}},
        )
        return [doc for doc, _ in similar]

    def add_event(
        self,
        event: str,
        template: str,
    ) -> None:
        """Add a new log event to the store, along with its template and associated graph data.

        The graph data will be inferred from the event and template.

        Args:
            event (str): The log event string to be added to the store.
            template (str): The template string associated with the log.

        """
        self.vector_store.add([event], metadatas=[{"template": template}])

        log_node = self.graph_store.query(
            f"""
            MATCH (l:{self.ontology.event_class_name} {{value: $value}})
            RETURN l
            """,
            params={"value": event},
        ).single()["l"]

        regex = re.sub(r"<<(.*?)>>", r"(?P<\1>.*?)", template)
        match = re.match(regex, event)

        nodes = [Node(uuid.uuid4(), key, {"value": match.group(key)}) for key in match.groupdict()]

        relationships = []
        for node in nodes:
            rel = self.ontology.get_event_object_property(node.type)
            if rel:
                relationships.append(Relationship(log_node, node, rel))

        self.graph_store.add_graph_documents(
            [GraphDocument(nodes=nodes, relationships=relationships, source=log_node)],
        )

    def __compose_similarity_question(self, log: str) -> str:
        return f"search_query: {log}"
