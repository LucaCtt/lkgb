from typing import Any

import neo4j
from langchain_neo4j import Neo4jGraph
from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship

from lkgb.config import Config


class Driver:
    """Graph store and vector index for the events knowledge graph.

    This class uses LangChain's Neo4jGraph api. It does not use the Neo4jVector api
    because it is not flexible enough.
    """

    def __init__(
        self,
        config: Config,
    ) -> "Driver":
        self.__config = config
        self.__graph_store = Neo4jGraph(
            url=config.neo4j_url,
            username=config.neo4j_username,
            password=config.neo4j_password,
            sanitize=True,
        )

    def initialize(self) -> None:
        """Initialize the graph store and the vector index.

        The graph store is initialized with the ontology and the labelled examples.
        The vector index is created for the event nodes, also populating the embeddings.
        Note: the ontology and examples data is not experiment-tagged.
        To reset it, the store must be cleared and re-initialized.
        This is intentional, as the ontology and examples are expected to be static among experiments.
        """
        # Get the latest experiment node
        latest_experiment = self.__graph_store.query(
            """MATCH (n:Experiment)
            RETURN elementID(n) as id,
                n.experiment_date_time as experimentDateTime,
                n.ontology_hash as ontologyHash,
                n.examples_hash as examplesHash
            ORDER BY experimentDateTime DESC
            LIMIT 1
            """,
        )
        if latest_experiment:
            if latest_experiment[0]["ontologyHash"] != self.__config.ontology_hash():
                msg = "The ontology has changed since the last experiment."
                raise ValueError(msg)

            if latest_experiment[0]["examplesHash"] != self.__config.examples_hash():
                msg = "The examples have changed since the last experiment."
                raise ValueError(msg)

            self.__graph_store.query(
                """
                MATCH (m:Experiment)
                WHERE elementID(m) = $id
                CREATE (n:Experiment $details)-[:SUBSEQUENT]->(m)
                """,
                params={"details": self.__config.dump(), "id": latest_experiment[0]["id"]},
            )
        else:
            # Create the experiment node
            self.__graph_store.query(
                """
                CREATE (n:Experiment $details)
                """,
                params={"details": self.__config.dump()},
            )

    def query(self, query: str, params: dict | None = None) -> list[dict[str, Any]]:
        if params is None:
            params = {}
        return self.__graph_store.query(query, params)

    def clear(self) -> None:
        """Clear any experiment in the graph store."""
        self.__graph_store.query("MATCH (n:Experiment) DETACH DELETE n")

    def get_subgraph_from_node(self, node_uri: str) -> GraphDocument:
        """Get the subgraph of a node in the store.

        The subgraph will contain all the nodes and relationships connected to the given node, even indirectly.
        """
        # Ugly but quite efficient. Also filters out the embedding property and the Resource label.
        nodes_subgraphs = self.__graph_store.query(
            """
            MATCH (n {uri: $node_uri})
            CALL apoc.path.subgraphAll(n, {})
            YIELD nodes, relationships
            RETURN
            [node IN nodes | {
                uri: node.uri,
                type: HEAD([label IN LABELS(node) WHERE label <> 'Resource']),
                properties: PROPERTIES(node)
            }] AS nodes,
            [rel IN relationships | {
                source: STARTNODE(rel).uri,
                target: ENDNODE(rel).uri,
                type: TYPE(rel)
            }] AS relationships
            """,
            params={"node_uri": node_uri},
        )

        if not nodes_subgraphs:
            return GraphDocument(nodes=[], relationships=[])

        nodes_subgraph = nodes_subgraphs[0]

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

        relationships = (
            [
                Relationship(
                    source=nodes_dict[relationship["source"]],
                    target=nodes_dict[relationship["target"]],
                    type=relationship["type"],
                )
                for relationship in nodes_subgraph["relationships"]
            ]
            if "relationships" in nodes_subgraph
            else []  # The node may not have any relationships
        )

        return GraphDocument(
            nodes=list(nodes_dict.values()),
            relationships=relationships,
        )
