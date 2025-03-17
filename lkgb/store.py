from pathlib import Path

from langchain_neo4j import Neo4jGraph

TIME_ONTOLOGY_URL = "http://www.w3.org/2006/time"
ONTOLOGY_PATH = "ontologies/log.ttl"


class OntologyStore:
    def __init__(
        self,
        url: str,
        username: str,
        password: str,
    ) -> "OntologyStore":
        self.graph_store = Neo4jGraph(url=url, username=username, password=password)

        self.__initialize_store()

        """
        self.vector_store = Neo4jVector.from_existing_graph(
            url=url,
            username=username,
            password=password,
            embedding=embeddings_model,
            node_label=resource_class_name,
            text_node_properties=["uri"],
            embedding_node_property="embedding",
        )
        """

    def __initialize_store(self) -> None:
        # Check if the database contains any Resource nodes,
        # if there are then the store is already initialized
        result = self.graph_store.query(
            """
            MATCH (n:Resource)
            RETURN COUNT(n) AS count
            """,
        )
        if result[0]["count"] != 0:
            return

        # Check if the neosemantics configuration is present,
        # if not, initialize it
        result = self.graph_store.query(
            """
            MATCH (n:_GraphConfig)
            RETURN COUNT(n) AS count
            """,
        )
        if result[0]["count"] == 0:
            self.graph_store.query(
                """
                CALL n10s.graphconfig.init();
                CALL n10s.graphconfig.set({ handleVocabUris: "IGNORE" });
                """,
            )

        # Check if uniqueness constraint is present,
        # if not, initialize it
        result = self.graph_store.query(
            """
            CALL db.constraints() YIELD name
            WHERE name = 'n10s_unique_uri'
            RETURN COUNT(*) AS count
            """,
        )
        if result[0]["count"] == 0:
            self.graph_store.query(
                """
                CREATE CONSTRAINT n10s_unique_uri ON (r:Resource) ASSERT r.uri IS UNIQUE;
                """,
            )

        with Path(ONTOLOGY_PATH).open() as f:
            ontology = f.read()
            # Load the log ontology
            self.graph_store.query(
                f"""
                WITH '{ontology}' AS ttl
                CALL n10s.onto.import.inline(ttl, "Turtle",);
                CALL n10s.onto.import.fetch("{TIME_ONTOLOGY_URL}", "Turtle");
                """,
            )

    def clear(self) -> None:
        self.graph_store.query(
            """
            MATCH (n:Resource) DETACH DELETE n;
            MATCH (n:_GraphConfig) DETACH DELETE n;
            CALL db.constraints.drop('n10s_unique_uri');
            """,
        )

    def triples(self) -> list[str, str, str]:
        triples = self.graph_store.query(
            """
            MATCH (n:Resource)<-[:DOMAIN]-(r:Relationship)-[:RANGE]->(m:Resource)
            WHERE n.uri STARTS WITH 'https://w3id.org/lkgb'
            AND m.uri STARTS WITH 'https://w3id.org/lkgb'
            AND r.uri STARTS WITH 'https://w3id.org/lkgb'
            RETURN n.name AS subject, r.name AS predicate, m.name AS object
            UNION
            MATCH (n:Class)<-[:DOMAIN]-(r:Property)-[:RANGE]->(m:Resource)
            WHERE n.uri = 'http://www.w3.org/2006/time#Instant'
            AND m.uri STARTS WITH 'http://www.w3.org/2001/XMLSchema'
            RETURN n.name AS subject, r.name AS predicate, m.name AS object
            """,
        )
        return [(row["subject"], row["predicate"], row["object"]) for row in triples]
