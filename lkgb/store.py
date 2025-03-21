"""Store for the events knowledge graph."""

from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship

LOG_ONTOLOGY_URL = "http://example.com/logs/dictionary"
LOG_EXAMPLES_URL = "http://example.com/logs/examples"
TIME_ONTOLOGY_URL = "http://www.w3.org/2006/time"
EVENTS_INDEX_NAME = "eventMessageIndex"


class EventsStore:
    """Store for the events knowledge graph."""

    def __init__(
        self,
        url: str,
        username: str,
        password: str,
        embeddings: Embeddings,
        experiment_id: str,
    ) -> "EventsStore":
        self.url = url
        self.username = username
        self.password = password
        self.embeddings = embeddings
        self.experiment_id = experiment_id

        self.graph_store = Neo4jGraph(url=url, username=username, password=password)

        # The creation of the vector index is deferred until the store is initialized,
        # so the embeddings are automatically computed for the nodes.
        self.vector_index: Neo4jVector = None

    def initialize(self, ontology_path: str, examples_path: str) -> None:
        """Initialize the store with the ontology and examples."""
        # Check if the neosemantics configuration is present,
        # if not, initialize it
        result = self.graph_store.query("MATCH (n:_GraphConfig) RETURN COUNT(n) AS count")
        if result[0]["count"] == 0:
            self.graph_store.query("CALL n10s.graphconfig.init()")
            self.graph_store.query("CALL n10s.graphconfig.set({ handleVocabUris: 'IGNORE' })")

        # Check if uniqueness constraint is present,
        # if not, initialize it
        result = self.graph_store.query(
            """
            SHOW CONSTRAINTS YIELD name
            WHERE name = 'n10s_unique_uri'
            RETURN COUNT(*) AS count
            """,
        )
        if result[0]["count"] == 0:
            self.graph_store.query("CREATE CONSTRAINT n10s_unique_uri FOR (r:Resource) REQUIRE r.uri IS UNIQUE")

        # Load the ontologies
        ontology = Path(ontology_path).read_text()
        self.graph_store.query(
            "CALL n10s.onto.import.inline($ontology, 'Turtle')",
            params={"ontology": ontology},
        )
        self.graph_store.query(
            "CALL n10s.onto.import.fetch($url, 'Turtle')",
            params={"url": TIME_ONTOLOGY_URL},
        )

        # Load the labelled examples
        examples = Path(examples_path).read_text()
        self.graph_store.query(
            "CALL n10s.rdf.import.inline($examples, 'Turtle')",
            params={"examples": examples},
        )

        # Add the experiment_id property to all nodes
        self.graph_store.query(
            "MATCH (n) SET n.experiment_id = $experiment_id",
            params={"experiment_id": self.experiment_id},
        )

        # Create the vector index
        self.vector_index = Neo4jVector.from_existing_graph(
            url=self.url,
            username=self.username,
            password=self.password,
            embedding=self.embeddings,
            node_label="Event",
            index_name=EVENTS_INDEX_NAME,
            text_node_properties=["message"],
            embedding_node_property="embedding",
        )

    def clear(self) -> None:
        """Clear the store to its initial state."""
        self.graph_store.query("MATCH (n) DETACH DELETE n")
        self.graph_store.query("DROP INDEX $index_name", params={"index_name": EVENTS_INDEX_NAME})

    def ontology_graph(self) -> GraphDocument:
        """Return the ontology graph as a GraphDocument.

        Returns:
            GraphDocument: The ontology graph, where nodes are classes
            and relationships are relationships between classes.

        """
        nodes_with_props = self.graph_store.query(
            """
            MATCH (c:Class)
            WHERE c.uri STARTS WITH $log_ontology_url OR c.uri = $time_instant_url
            OPTIONAL MATCH (c)<-[:DOMAIN]-(p:Property)
            WITH c.name AS class, elementID(c) as id, COLLECT([p.name, p.comment]) AS pairs
            RETURN class, id, apoc.map.fromPairs(pairs) AS properties
            """,
            params={
                "log_ontology_url": LOG_ONTOLOGY_URL,
                "time_instant_url": f"{TIME_ONTOLOGY_URL}#Instant",
            },
        )
        nodes_dict = {
            row["id"]: Node(id=row["id"], type=row["class"], properties=row["properties"]) for row in nodes_with_props
        }

        triples = self.graph_store.query(
            """
            MATCH (n:Class)<-[:DOMAIN]-(r:Relationship)-[:RANGE]->(m:Class)
            WHERE n.uri STARTS WITH $log_ontology_url
            AND m.uri STARTS WITH $log_ontology_url
            AND r.uri STARTS WITH $log_ontology_url
            RETURN elementID(n) AS subject_id, r.name AS predicate, elementID(m) AS object_id
            """,
            params={"log_ontology_url": LOG_ONTOLOGY_URL},
        )
        relationships = [
            Relationship(
                source=nodes_dict[row["subject_id"]],
                target=nodes_dict[row["object_id"]],
                type=row["predicate"],
            )
            for row in triples
        ]

        return GraphDocument(
            nodes=list(nodes_dict.values()),
            relationships=relationships,
        )

    def add_event_graph(self, graph: GraphDocument) -> None:
        """Add an event graph to the store.

        Args:
            graph (GraphDocument): The event graph to add.

        """
        for node in graph.nodes:
            additional_properties = {"experiment_id": self.experiment_id}
            if node.type == "Event":
                additional_properties["embedding"] = self.embeddings.embed_query(node.properties["message"])

            self.graph_store.query(
                f"""
                CREATE (n:{node.type})
                SET n += $props
                """,
                params={"props": node.properties + additional_properties},
            )

        for relationship in graph.relationships:
            self.graph_store.query(
                f"""
                MATCH (a:{relationship.source.type}),(b:{relationship.target.type})
                WHERE elementID(a) = $source_id AND elementID(b) = $target_id
                CREATE (a)-[r:{relationship.type}]->(b)
                """,
                params={"source_id": relationship.source.id, "target_id": relationship.target.id},
            )

    def search_similar_events(self, event: str, k: int = 3) -> GraphDocument:
        """Search for similar events in the store.

        Args:
            event (str): The event message to search for.
            k (int): The number of similar events to return.

        Returns:
            GraphDocument: The graph of similar events and their relationships.

        """
        query_embeddings = self.embeddings.embed_query(event)

        result = self.graph_store.query(
            """
            CALL db.index.vector.queryNodes($events_index_name, $k, $query_embeddings)
            YIELD node as similarEvent, score
            OPTIONAL MATCH (similarEvent)-[r]-(connected)
            WITH collect(DISTINCT similarEvent) + collect(DISTINCT connected) AS allNodes,
                collect(DISTINCT r) AS allRels
            RETURN allNodes, allRels
            """,
            params={"events_index_name": EVENTS_INDEX_NAME, "k": k, "query_embeddings": query_embeddings},
        )

        nodes_dict = {
            node.id: Node(
                id=node.id,
                type=node.labels[0],
                properties=dict(node),
            )
            for node in result[0]["allNodes"]
        }

        relationships = [
            Relationship(
                source=nodes_dict[rel.startNodeElementId],
                target=nodes_dict[rel.endNodeElementId],
                type=rel.type,
            )
            for rel in result[0]["allRels"]
        ]

        return GraphDocument(
            nodes=list(nodes_dict.values()),
            relationships=relationships,
        )
