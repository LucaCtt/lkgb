from typing import cast

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship
from pydantic import BaseModel, Field

from lkgb.reports import ParserReport
from lkgb.store import OntologyStore
from lkgb.tools import fetch_ip_address_info


class _EventGraph(BaseModel):
    nodes: list
    relationships: list


system_prompt = (
    "## 1. Overview\n"
    "You are a top-tier algorithm designed for extracting information from a log event "
    "in structured formats to build a knowledge graph according to an ontology.\n"
    "Try to capture as much information from the event as possible without sacrificing accuracy. "
    "Do not add any information that is not explicitly mentioned in the event.\n"
    "- The aim is to achieve exhaustiveness in the knowledge graph, making it ontology-compliant.\n"
    "- **IDs**: IDs must be unique for each node and consistent when referenced."
    "## 2. Context Enrichment\n"
    "Use the provided tool to retrieve additional information about IP addresses you find.\n"
    "Use the additional information to enrich the knowledge graph.\n"
    "## 3. Strict Compliance\n"
    "Adhere to the rules strictly. Non-compliance will result in termination."
)

gen_graph_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        ("placeholder", "{examples}"),
        ("human", "Event: '{log}'"),
        ("placeholder", "{messages}"),
    ],
)


def _get_example_group(event: str, nodes: list[dict], relationships: dict) -> list[BaseMessage]:
    return [
        HumanMessage(f"Event: '{event}", name="example_user"),
        AIMessage(
            "",
            name="example_assistant",
            tool_calls=[
                {
                    "name": "DynamicEventGraph",
                    "args": {
                        "nodes": nodes,
                        "relationships": relationships,
                    },
                    "id": "1",
                },
            ],
        ),
        ToolMessage("", tool_call_id="1"),
    ]


class Parser:
    """The Parser class is responsible for parsing log events and identifying their templates."""

    def __init__(
        self,
        parser_model: BaseLanguageModel,
        ontology: OntologyStore,
        self_reflection_steps: int,
    ) -> "Parser":
        self.history = ChatMessageHistory()
        self.ontology = ontology
        self.self_reflection_steps = self_reflection_steps

        try:
            parser_model.with_structured_output(_EventGraph)
        except NotImplementedError as e:
            msg = "The parser model must support structured output."
            raise ValueError(msg) from e

        # Add context enrichment tools.
        # Note: not all models support tools + structured output
        structured_model = parser_model.bind_tools([fetch_ip_address_info])

        # Add the graph structure to the structured output.
        # Also include raw output to retrieve eventual errors.
        structured_model = structured_model.with_structured_output(self.__create_graph_structure(), include_raw=True)

        # TODO: this does not work with structured output
        self.chain = RunnableWithMessageHistory(
            gen_graph_prompt | structured_model,
            lambda _: self.history,
            input_messages_key="log",
            history_messages_key="messages",
        )

    def __create_graph_structure(self) -> type[_EventGraph]:
        onto = self.ontology.graph()

        valid_node_types = [node.type for node in onto.nodes]
        valid_relationships = [rel.type for rel in onto.relationships]

        valid_properties = []
        for node in onto.nodes:
            valid_properties = valid_properties + list(node.properties.keys())

        class _Property(BaseModel):
            key: str = Field(
                description=f"Property key. Available types with their allow type are: {valid_properties}",
                enum=valid_properties,
            )
            value: str = Field(description=("Extracted value."))

        class _Node(BaseModel):
            id: str = Field(description="Name or human-readable unique identifier.")
            type: str = Field(
                description=f"The type or label of the node. Available options are: {valid_node_types}",
                enum=valid_node_types,
            )
            properties: list[_Property | None] = Field(None, description="List of node properties.")

        class _Relationship(BaseModel):
            source_id: str = Field(description="Name or human-readable unique identifier of source node.")
            target_id: str = Field(description="Name or human-readable unique identifier of source node.")
            type: str = Field(
                description=f"The type or label of the relationship. Available types are: {valid_relationships}",
                enum=valid_relationships,
            )

        class DynamicEventGraph(_EventGraph):
            nodes: list[_Node] = Field(description="List of nodes.")
            relationships: list[_Relationship] = Field(
                description="List of relationships.",
            )

        return DynamicEventGraph

    def _get_examples(self) -> list[BaseMessage]:
        # TODO: Move this example to the graph store(?)
        return _get_example_group(
            "2022-01-21 00:09:11 jhall/192.168.230.165:46011 TLS: soft reset sec=3308/3308 bytes=45748/-1 pkts=649/0",
            nodes=[
                {
                    "id": "1",
                    "type": "Event",
                    "properties": [
                        {"key": "message", "value": "TLS: soft reset sec=3308/3308 bytes=45748/-1 pkts=649/0"},
                    ],
                },
                {"id": "2", "type": "User", "properties": [{"key": "username", "value": "jhall"}]},
                {
                    "id": "3",
                    "type": "Address",
                    "properties": [
                        {"key": "ipv4", "value": "192.168.230.165"},
                        {"key": "port", "value": "46011"},
                        {"key": "city", "value": "Los Angeles"},
                        {"key": "region", "value": "California"},
                        {"key": "country", "value": "United States"},
                        {"key": "timezone", "value": "+08:00"},
                        {"key": "asn", "value": "AS1234"},
                        {"key": "organization", "value": "Example Inc."},
                    ],
                },
                {
                    "id": "4",
                    "type": "TimeStamp",
                    "properties": [
                        {"key": "inXSDDateTime", "value": "2022-01-21 00:09:11"},
                    ],
                },
            ],
            relationships=[
                {"source_id": "1", "target_id": "2", "type": "hasUser"},
                {"source_id": "1", "target_id": "3", "type": "hasAddress"},
                {"source_id": "1", "target_id": "4", "type": "hasTimeStamp"},
            ],
        )

    def parse(self, log: str) -> ParserReport:
        """Given a log, this function identifies and updates the template for the log and also for similar logs.

        It first searches for very similar logs in the store and checks if their templates match the current log.
        If no matching template is found, it searches for sufficiently similar logs
        and uses them to generate a template.

        Args:
            log (str): The log for weich the template needs to be identified.

        Returns:
            ParserReport: A ParserReport object containing the timings of the parsing operations.

        """
        report = ParserReport()

        raw_schema = self.chain.invoke(
            {"log": log, "examples": self._get_examples()},
            {"configurable": {"session_id": "unused"}},
        )

        raw_schema = cast(dict, raw_schema)

        nodes_dict = {
            node.id: Node(
                id=node.id,
                type=node.type,
                properties={prop.key: prop.value for prop in node.properties} if node.properties else {},
            )
            for node in raw_schema["parsed"].nodes
        }
        relationships = []
        for rel in raw_schema["parsed"].relationships:
            source_node = nodes_dict[rel.source_id]
            target_node = nodes_dict[rel.target_id]

            relationships.append(Relationship(source=source_node, target=target_node, type=rel.type))

        graph = GraphDocument(nodes=list(nodes_dict.values()), relationships=relationships)

        return report.finish(graph)
