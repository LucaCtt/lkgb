from pathlib import Path

from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship

from lkgb.config import Config
from lkgb.store.driver import Driver

LOG_ONTOLOGY_URL = "http://example.com/lkgb/logs/dictionary"
TIME_ONTOLOGY_URL = "http://www.w3.org/2006/time"
N10S_CONSTRAINT_NAME = "n10s_unique_uri"


class Ontology:
    def __init__(self, driver: Driver) -> "Ontology":
        self.driver = driver

    def initialize(self, config: Config) -> None:
        # Check if the neosemantics configuration is present,
        # if it is, assume the ontology and examples are already loaded.
        result = self.driver.query("MATCH (n:_GraphConfig) RETURN COUNT(n) AS count")
        if result[0]["count"] != 0:
            return

        # Init neosemantics plugin
        self.driver.query("CALL n10s.graphconfig.init()")
        self.driver.query("CALL n10s.graphconfig.set({ handleVocabUris: 'IGNORE' })")
        self.driver.query(f"CREATE CONSTRAINT {N10S_CONSTRAINT_NAME} FOR (r:Resource) REQUIRE r.uri IS UNIQUE")

        # Load the ontologies
        self.driver.query(
            "CALL n10s.onto.import.inline($ontology, 'Turtle')",
            params={"ontology": Path(config.ontology_path).read_text()},
        )
        self.driver.query(
            "CALL n10s.onto.import.fetch($url, 'Turtle')",
            params={"url": TIME_ONTOLOGY_URL},
        )

    def clear(self) -> None:
        """Clear the store to its initial state."""
        self.driver.query(
            "MATCH (n:Resource) WHERE n.uri STARTS WITH $time_url OR n.uri STARTS WITH $log_url DETACH DELETE n",
            params={"time_url": TIME_ONTOLOGY_URL, "log_url": LOG_ONTOLOGY_URL},
        )
        self.driver.query(
            "DROP CONSTRAINT $constraint_name IF EXISTS",
            params={"constraint_name": N10S_CONSTRAINT_NAME},
        )

    def ontology_graph(self) -> GraphDocument:
        """Return the ontology graph as a GraphDocument.

        The returned nodes and relationship types will be without uris. This may not be the best idea,
        only time will tell.

        Note that this will not return all of the classes and relationships from external ontologies,
        but only the relevant ones for this project.

        Returns:
            GraphDocument: The ontology graph, where nodes are classes
            and relationships are relationships between classes.

        """
        nodes_with_props = self.driver.query(
            """
            MATCH (c:Class)
            WHERE c.uri STARTS WITH $log_ontology_url OR c.uri = $time_instant_url
            OPTIONAL MATCH (c)<-[:DOMAIN]-(p:Property)
            WITH c.name AS class, c.uri as uri, COLLECT([p.name, p.comment]) AS pairs
            RETURN class, uri, apoc.map.fromPairs(pairs) AS properties
            """,
            params={
                "log_ontology_url": LOG_ONTOLOGY_URL,
                "time_instant_url": f"{TIME_ONTOLOGY_URL}#Instant",
            },
        )
        nodes_dict = {
            row["uri"]: Node(id=row["uri"], type=row["class"], properties=row["properties"]) for row in nodes_with_props
        }

        triples = self.driver.query(
            """
            MATCH (n:Class)<-[:DOMAIN]-(r:Relationship)-[:RANGE]->(m:Class)
            WHERE n.uri STARTS WITH $log_ontology_url
            AND m.uri STARTS WITH $log_ontology_url
            AND r.uri STARTS WITH $log_ontology_url
            RETURN n.uri AS subject_uri, r.name AS predicate, m.uri AS object_uri
            """,
            params={"log_ontology_url": LOG_ONTOLOGY_URL},
        )
        relationships = [
            Relationship(
                source=nodes_dict[row["subject_uri"]],
                target=nodes_dict[row["object_uri"]],
                type=row["predicate"],
            )
            for row in triples
        ]

        return GraphDocument(
            nodes=list(nodes_dict.values()),
            relationships=relationships,
        )
