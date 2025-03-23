"""Parser module for parsing log events and constructing knowledge graphs."""

import uuid
from typing import cast

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship
from pydantic import BaseModel, Field

from lkgb.reports import ParserReport
from lkgb.store import EventsStore
from lkgb.tools import fetch_ip_address_info


class _EventGraph(BaseModel):
    nodes: list
    relationships: list


def _get_example_group(event: str, graph: GraphDocument) -> list[BaseMessage]:
    nodes = [
        {
            "id": node.id,
            "type": node.type,
            "properties": [{"type": key, "value": value} for key, value in node.properties.items()],
        }
        for node in graph.nodes
    ]

    relationships = [
        {
            "source_id": rel.source.id,
            "target_id": rel.target.id,
            "type": rel.type,
        }
        for rel in graph.relationships
    ]

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
        store: EventsStore,
        prompt_build_graph: str,
        self_reflection_steps: int,
    ) -> "Parser":
        self.history = ChatMessageHistory()
        self.store = store
        self.prompt_build_graph = prompt_build_graph
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

        gen_graph_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.prompt_build_graph,
                ),
                ("placeholder", "{examples}"),
                ("human", "Event: '{event}'"),
                ("placeholder", "{messages}"),
            ],
        )

        self.chain = gen_graph_prompt | structured_model

    def __create_graph_structure(self) -> type[_EventGraph]:
        ontology = self.store.ontology_graph()

        valid_node_types = [node.type for node in ontology.nodes]
        valid_relationships = [rel.type for rel in ontology.relationships]

        valid_properties = []
        for node in ontology.nodes:
            valid_properties = valid_properties + list(node.properties.keys())

        class _Property(BaseModel):
            type: str = Field(
                description=f"The type or label of the propertye. Available options are: {valid_properties}",
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

    def _get_examples(self, event: str) -> list[BaseMessage]:
        similar_event_graph = self.store.search_similar_events(event, k=1)
        return _get_example_group(event, similar_event_graph[0])

    def parse(self, event: str) -> ParserReport:
        """Parse the given event and construct a knowledge graph.

        Args:
            event: The log event to parse.

        Returns:
            A report containing the stats of the parsing process.

        """
        report = ParserReport()

        raw_schema = self.chain.invoke(
            {"event": event, "examples": self._get_examples(event)},
            {"configurable": {"session_id": "unused"}},
        )

        raw_schema = cast(dict, raw_schema)

        nodes_dict = {
            node.id: Node(
                id=str(uuid.uuid4()),
                type=node.type,
                properties={prop.type: prop.value for prop in node.properties} if node.properties else {},
            )
            for node in raw_schema["parsed"].nodes
        }
        relationships = []
        for rel in raw_schema["parsed"].relationships:
            source_node = nodes_dict[rel.source_id]
            target_node = nodes_dict[rel.target_id]

            relationships.append(Relationship(source=source_node, target=target_node, type=rel.type))

        graph = GraphDocument(nodes=list(nodes_dict.values()), relationships=relationships)
        self.store.add_event_graph(graph)

        return report.finish()
