from pathlib import Path

from langchain_neo4j import Neo4jGraph
from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship

TIME_ONTOLOGY_URL = "http://www.w3.org/2006/time"
ONTOLOGY_PATH = "ontologies/logs.ttl"


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
                "CALL n10s.graphconfig.init()",
            )
            self.graph_store.query(
                "CALL n10s.graphconfig.set({ handleVocabUris: 'IGNORE' })",
            )

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
            self.graph_store.query(
                """
                CREATE CONSTRAINT n10s_unique_uri FOR (r:Resource) REQUIRE r.uri IS UNIQUE;
                """,
            )

        ontology = Path(ONTOLOGY_PATH).read_text()
        # Load the log ontology
        self.graph_store.query(
            f"""
            CALL n10s.onto.import.inline('{ontology}', "Turtle")
            """,
        )
        self.graph_store.query(f"CALL n10s.onto.import.fetch('{TIME_ONTOLOGY_URL}', 'Turtle')")

    def clear(self) -> None:
        self.graph_store.query("MATCH (n:Resource) DETACH DELETE n")
        self.graph_store.query("MATCH (n:_GraphConfig) DETACH DELETE n")
        self.graph_store.query("DROP CONSTRAINT n10s_unique_uri")

    def graph(self) -> GraphDocument:
        nodes_with_props = self.graph_store.query(
            """
            MATCH (c:Class)
            WHERE c.uri STARTS WITH 'https://w3id.org/lkgb' OR c.uri = 'http://www.w3.org/2006/time#Instant'
            OPTIONAL MATCH (c)<-[:DOMAIN]-(p:Property)
            WITH c.name AS class, elementID(c) as id, COLLECT([p.name, p.comment]) AS pairs
            RETURN class, id, apoc.map.fromPairs(pairs) AS properties
            """,
        )
        nodes_dict = {
            row["id"]: Node(id=row["id"], type=row["class"], properties=row["properties"]) for row in nodes_with_props
        }

        triples = self.graph_store.query(
            """
            MATCH (n:Class)<-[:DOMAIN]-(r:Relationship)-[:RANGE]->(m:Class)
            WHERE n.uri STARTS WITH 'https://w3id.org/lkgb'
            AND m.uri STARTS WITH 'https://w3id.org/lkgb'
            AND r.uri STARTS WITH 'https://w3id.org/lkgb'
            RETURN elementID(n) AS subject_id, r.name AS predicate, elementID(m) AS object_id
            """,
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
