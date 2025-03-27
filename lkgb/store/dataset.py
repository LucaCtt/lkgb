from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_neo4j.graphs.graph_document import GraphDocument

from lkgb.config import Config
from lkgb.store.driver import Driver
from lkgb.store.module import StoreModule

EVENTS_INDEX_NAME = "eventMessageIndex"
LOG_EXAMPLES_URL = "http://example.com/lkgb/logs/examples"
LOG_TESTS_URL = "http://example.com/lkgb/logs/tests"


class TestEvent:
    """A test event with its context and ground truth."""

    def __init__(self, event: str, context: dict, ground_truth: GraphDocument) -> None:
        self.event = event
        self.context = context
        self.ground_truth = ground_truth


class Dataset(StoreModule):
    """The Dataset module is responsible for managing the event graphs in the store.

    Includes the loading of the examples and tests, and the search for similar events.
    """

    def __init__(self, config: Config, driver: Driver, embeddings: Embeddings) -> None:
        super().__init__(config)

        self.__driver = driver
        self.__embeddings = embeddings

    def initialize(self) -> None:
        # Check if the examples are already loaded
        result = self.__driver.query(
            """
            MATCH (n:Resource) WHERE n.uri STARTS WITH $examples_uri RETURN COUNT(n) AS count
            """,
            params={"examples_uri": LOG_EXAMPLES_URL},
        )
        if result[0]["count"] != 0:
            return

        # Load the examples
        self.__driver.query(
            "CALL n10s.rdf.import.inline($examples, 'Turtle')",
            params={"examples": Path(self._config.examples_path).read_text()},
        )

        # Create the vector index
        self.__driver.query(
            f"""
            CREATE VECTOR INDEX {EVENTS_INDEX_NAME}
            FOR (n:Event) ON n.embedding
            OPTIONS {{ indexConfig : {{
                `vector.similarity_function` : 'cosine'
            }} }}
            """,
        )

        # Populate the embeddings for the examples
        to_populate = self.__driver.query(
            """
            MATCH (n:Event)
            WHERE n.embedding IS null
            RETURN elementId(n) AS id, n.eventMessage as eventMessage
            """,
        )
        text_embeddings = self.__embeddings.embed_documents([el["eventMessage"] for el in to_populate])
        self.__driver.query(
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

        # Load the tests
        # Note: the test events should not have an embedding
        self.__driver.query(
            "CALL n10s.rdf.import.inline($tests, 'Turtle')",
            params={"tests": Path(self._config.tests_path).read_text()},
        )

    def clear(self) -> None:
        self.__driver.query(f"DROP INDEX {EVENTS_INDEX_NAME} IF EXISTS")
        self.__driver.query(
            """
            MATCH (n:Resource)
            WHERE n.uri STARTS WITH $examples_url OR n.uri STARTS WITH $tests_url
            DETACH DELETE n
            """,
            params={"examples_url": LOG_EXAMPLES_URL, "tests_url": LOG_TESTS_URL},
        )
        # TODO: Make this more specific
        self.__driver.query(
            """
            MATCH (n)
            DETACH DELETE n
            """,
        )

    def tests(self) -> list[TestEvent]:
        test_nodes = self.__driver.query(
            """
            MATCH (n:Event)
            WHERE n.uri STARTS WITH $log_tests_url
            RETURN n.eventMessage as message, n.uri as uri
            """,
            params={"log_tests_url": LOG_TESTS_URL},
        )
        tests = []
        for test in test_nodes:
            ground_truth = self.__driver.get_subgraph_from_node(test["uri"])

            source_node = next((node for node in ground_truth.nodes if node.type == "Source"), None)
            context = (
                {"source": source_node.properties["sourceName"], "device": source_node.properties["sourceDevice"]}
                if source_node
                else {}
            )
            tests.append(TestEvent(event=test["message"], context=context, ground_truth=ground_truth))

        return tests

    def add_event_graph(self, graph: GraphDocument) -> None:
        """Add an event graph to the store.

        All the nodes will be tagged with the current experiment id,
        and for Event nodes the embedding will be added.

        Args:
            graph (GraphDocument): The event graph to add.

        """
        for node in graph.nodes:
            # Add the experiment_id and (for the Event nodes) the embedding.
            additional_properties: dict[str, Any] = {"experimentId": self._config.experiment_id}
            if node.type == "Event":
                # This will raise an exception if the LLM produces an Event node without a message property.
                additional_properties["embedding"] = self.__embeddings.embed_query(node.properties["eventMessage"])

            self.__driver.query(
                "CALL apoc.create.node([$type], $props) YIELD node",
                params={"type": node.type, "props": {**node.properties, **additional_properties}},
            )

        for relationship in graph.relationships:
            self.__driver.query(
                """
                MATCH (a {uri: $source_uri}), (b {uri: $target_uri})
                CALL apoc.create.relationship(a, $type, {}, b) YIELD rel
                RETURN rel
                """,
                params={
                    "source_uri": relationship.source.id,
                    "target_uri": relationship.target.id,
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
        query_embeddings = self.__embeddings.embed_query(event)

        # TODO: fix: this also retrieves stuff with different experimentId
        # Find k similar events using embeddings
        similar_events = self.__driver.query(
            """
            CALL db.index.vector.queryNodes($index, $k, $embedding)
            YIELD node, score
            RETURN node.uri AS node_uri, score
            """,
            params={"index": EVENTS_INDEX_NAME, "k": k, "embedding": query_embeddings},
        )

        return [
            self.__driver.get_subgraph_from_node(similar_event["node_uri"], ["experimentId"])
            for similar_event in similar_events
        ]
