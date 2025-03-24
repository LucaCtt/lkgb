"""Store for the events knowledge graph."""

from pathlib import Path

import neo4j.time
from langchain_core.embeddings import Embeddings
from langchain_neo4j import Neo4jGraph
from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship

LOG_ONTOLOGY_URL = "http://example.com/logs/dictionary"
LOG_EXAMPLES_URL = "http://example.com/logs/examples"
TIME_ONTOLOGY_URL = "http://www.w3.org/2006/time"
EVENTS_INDEX_NAME = "eventMessageIndex"
N10S_CONSTRAINT_NAME = "n10s_unique_uri"


class EventsStore:
    """Graph store and vector index for the events knowledge graph.

    This class uses LangChain's Neo4jGraph api. It does not use the Neo4jVector api
    because it is not flexible enough.
    """

    def __init__(
        self,
        url: str,
        username: str,
        password: str,
        embeddings: Embeddings,
        experiment_id: str,
    ) -> "EventsStore":
        self.embeddings = embeddings
        self.experiment_id = experiment_id

        self.graph_store = Neo4jGraph(url=url, username=username, password=password)

    def initialize(self, config_dict: dict) -> None:
        """Initialize the graph store and the vector index.

        The graph store is initialized with the ontology and the labelled examples.
        The vector index is created for the event nodes, also populating the embeddings.
        Note: the ontology and examples data is not experiment-tagged.
        To reset it, the store must be cleared and re-initialized.
        This is intentional, as the ontology and examples are expected to be static among experiments.
        """
        # Get the latest experiment node
        latest_experiment = self.graph_store.query(
            """MATCH (n:Experiment)
            RETURN elementID(n) as id,
                n.experiment_date_time as experiment_date_time,
                n.ontology_hash as ontology,
                n.examples_hash as examples_hash
            ORDER BY experiment_date_time DESC
            LIMIT 1
            """,
        )
        if latest_experiment:
            if latest_experiment[0]["ontology_hash"] == config_dict["ontology_hash"]:
                msg = "The ontology has changed since the last experiment."
                raise ValueError(msg)

            if latest_experiment[0]["examples_hash"] == config_dict["examples_hash"]:
                msg = "The examples have changed since the last experiment."
                raise ValueError(msg)

            self.graph_store.query(
                """
                CREATE (n:Experiment $details)<-[:SUBSEQUENT]-(m:Experiment)
                WHERE elementID(m) = $id
                """,
                params={"details": config_dict, "id": latest_experiment[0]["id"]},
            )
        else:
            # Create the experiment node
            self.graph_store.query(
                """
                CREATE (n:Experiment $details)
                """,
                params={"details": config_dict},
            )

        # Check if the neosemantics configuration is present,
        # if it is, assume the ontology and examples are already loaded.
        result = self.graph_store.query("MATCH (n:_GraphConfig) RETURN COUNT(n) AS count")
        if result[0]["count"] != 0:
            return

        # Init neosemantics plugin
        self.graph_store.query("CALL n10s.graphconfig.init()")
        self.graph_store.query("CALL n10s.graphconfig.set({ handleVocabUris: 'IGNORE' })")
        self.graph_store.query(f"CREATE CONSTRAINT {N10S_CONSTRAINT_NAME} FOR (r:Resource) REQUIRE r.uri IS UNIQUE")

        # Load the ontologies
        self.graph_store.query(
            "CALL n10s.onto.import.inline($ontology, 'Turtle')",
            params={"ontology": Path(config_dict["ontology_path"]).read_text()},
        )
        self.graph_store.query(
            "CALL n10s.onto.import.fetch($url, 'Turtle')",
            params={"url": TIME_ONTOLOGY_URL},
        )

        # Load the examples
        self.graph_store.query(
            "CALL n10s.rdf.import.inline($examples, 'Turtle')",
            params={"examples": Path(config_dict["examples_path"]).read_text()},
        )

        # Create the vector index
        self.graph_store.query(
            f"CALL db.index.vector.createIndex('{EVENTS_INDEX_NAME}', 'Event', 'embedding', 'cosine')",
        )

        # Populate the embeddings for the examples
        to_populate = self.graph_store.query(
            """
            MATCH (n:Event)
            WHERE n.embedding IS null
            RETURN elementId(n) AS id, reduce(str='', k IN $props | str + '\\n' + k + ':' + coalesce(n[k], '')) AS text
            """,
        )
        text_embeddings = self.embeddings.embed_documents([el["text"] for el in to_populate])
        self.graph_store.query(
            """
            UNWIND $data AS row
            MATCH (n:Event)
            WHERE elementId(n) = row.id
            CALL db.create.setNodeVectorProperty(n, 'embedding', row.embedding)
            """,
            params={
                "data": [
                    {"id": el["id"], "embedding": embedding}
                    for el, embedding in zip(to_populate, text_embeddings, strict=True)
                ],
            },
        )

    def clear(self) -> None:
        """Clear the store to its initial state."""
        self.graph_store.query("MATCH (n) DETACH DELETE n")
        self.graph_store.query("DROP INDEX $index_name", params={"index_name": EVENTS_INDEX_NAME})
        self.graph_store.query("DROP CONSTRAINT $constraint_name", params={"constraint_name": N10S_CONSTRAINT_NAME})

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
        nodes_with_props = self.graph_store.query(
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

        triples = self.graph_store.query(
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

    def add_event_graph(self, graph: GraphDocument) -> None:
        """Add an event graph to the store.

        All the nodes will be tagged with the current experiment id,
        and for Event nodes the embedding will be added.

        Args:
            graph (GraphDocument): The event graph to add.

        """
        for node in graph.nodes:
            # Add the experiment_id and (for the Event nodes) the embedding.
            additional_properties = {"experiment_id": self.experiment_id}
            if node.type == "Event":
                # This will raise an exception if the LLM produces an Event node without a message property.
                additional_properties["embedding"] = self.embeddings.embed_query(node.properties["message"])

            self.graph_store.query(
                "CALL apoc.create.node([$type], $props) YIELD node",
                params={"type": node.type, "props": {**node.properties, **additional_properties}},
            )

        for relationship in graph.relationships:
            self.graph_store.query(
                """
                MATCH (a {uri: $source_uri}), (b {uri: $target_uri})
                CALL apoc.create.relationship(a, $type, {}, b) YIELD rel
                """,
                params={
                    "source_uri": relationship.source.uri,
                    "target_uri": relationship.target.uri,
                    "type": relationship.type,
                },
            )

    def search_similar_events(self, event: str, k: int = 5) -> list[GraphDocument]:
        """Search for similar events in the store.

        Args:
            event (str): The event message to search for.
            k (int): The number of similar events to search for.

        Returns:
            list[GraphDocument]: The list of graphs of similar events,
                with the nodes they are connected to and their relationships.

        """
        query_embeddings = self.embeddings.embed_query(event)

        # Find k similar events using embeddings
        similar_events = self.graph_store.query(
            """
            CALL db.index.vector.queryNodes($index, $k, $embedding)
            YIELD node, score
            RETURN node.uri AS node_uri, score
            """,
            params={"index": EVENTS_INDEX_NAME, "k": k, "embedding": query_embeddings},
        )

        return [self.__get_subgraph_from_node(similar_event["node_uri"]) for similar_event in similar_events]

    def __get_subgraph_from_node(self, node_uri: str) -> GraphDocument:
        """Get the subgraph of a node in the store.

        The subgraph will contain all the nodes and relationships connected to the given node, even indirectly.
        """
        # Ugly but quite efficient. Also filters out the embedding property and the Resource label.
        nodes_subgraph = self.graph_store.query(
            """
            MATCH (n {uri: $node_uri})
            CALL apoc.path.subgraphAll(n, {})
            YIELD nodes, relationships
            RETURN
            [node IN nodes | {
                uri: node.uri,
                type: HEAD([label IN LABELS(node) WHERE label <> 'Resource']),
                properties: apoc.map.removeKey(PROPERTIES(node), 'embedding')
            }] AS nodes,
            [rel IN relationships | {
                source: STARTNODE(rel).uri,
                target: ENDNODE(rel).uri,
                type: TYPE(rel)
            }] AS relationships
            """,
            params={"node_uri": node_uri},
        )[0]  # Only one row is returned

        # The neo4j date and time objects are quite problematic, as they are not JSON serializable.
        # This is a workaround to convert them to strings.
        for node in nodes_subgraph["nodes"]:
            for key, value in node["properties"].items():
                if isinstance(value, neo4j.time.DateTime):
                    node["properties"][key] = value.iso_format()
                if isinstance(value, neo4j.time.Date):
                    node["properties"][key] = value.iso_format()

        nodes_dict = {
            node["uri"]: Node(id=node["uri"], type=node["type"], properties=node["properties"])
            for node in nodes_subgraph["nodes"]
        }

        relationships = [
            Relationship(
                source=nodes_dict[relationship["source"]],
                target=nodes_dict[relationship["target"]],
                type=relationship["type"],
            )
            for relationship in nodes_subgraph["relationships"]
        ]

        return GraphDocument(
            nodes=list(nodes_dict.values()),
            relationships=relationships,
        )
