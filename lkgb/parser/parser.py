"""Parser module for parsing log events and constructing knowledge graphs."""

import uuid
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j.graphs.graph_document import GraphDocument

from lkgb.parser.models import EventGraph, build_dynamic_model
from lkgb.parser.reports import ParserReport
from lkgb.store import Store
from lkgb.tools import fetch_ip_address_info


def _get_message_group(event: str, graph: GraphDocument, context: dict) -> list[BaseMessage]:
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

    tool_call_id = f"call_{uuid.uuid4()!s}"

    return [
        HumanMessage(f"Event: '{event}'\nContext: {context}", name="example_user"),
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
                    "id": tool_call_id,
                },
            ],
        ),
        ToolMessage("", tool_call_id=tool_call_id),
    ]


class Parser:
    """The Parser class is responsible for parsing log events and identifying their templates."""

    def __init__(
        self,
        parser_model: BaseChatModel,
        store: Store,
        prompt_build_graph: str,
        self_reflection_steps: int,
    ) -> None:
        self.store = store
        self.prompt_build_graph = prompt_build_graph
        self.self_reflection_steps = self_reflection_steps

        try:
            parser_model.with_structured_output(EventGraph)
        except NotImplementedError as e:
            msg = "The parser model must support structured output."
            raise ValueError(msg) from e

        # Add context enrichment tools.
        # Note: not all models support tools + structured output
        structured_model = parser_model.bind_tools([fetch_ip_address_info])

        # Add the graph structure to the structured output.
        # Also include raw output to retrieve eventual errors.
        structured_model = structured_model.with_structured_output(  # type: ignore[attr-defined]
            build_dynamic_model(store.ontology.graph()),
            include_raw=True,
        )

        gen_graph_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompt_build_graph),
                ("placeholder", "{examples}"),
                ("human", "Event: '{event}'\nContext: '{context}'"),
            ],
        )

        self.chain = gen_graph_prompt | structured_model

    def _get_examples(self, event: str) -> list[BaseMessage]:
        similar_events = self.store.dataset.events_mmr_search(event, k=2)

        messages = []
        for similar_event, graph in similar_events:
            source_node = next((node for node in graph.nodes if node.type == "Source"), None)

            context = {
                key: source_node.properties[key]
                for key in ["sourceName", "sourceType", "sourceDevice"]
                if source_node and key in source_node.properties
            }

            messages.extend(_get_message_group(similar_event, graph, context))

        return messages

    def parse(self, event: str, context: dict) -> ParserReport:
        """Parse the given event and construct a knowledge graph.

        Args:
            event: The log event to parse.
            context: The context of the event.

        Returns:
            A report containing the stats of the parsing process.

        """
        report = ParserReport()

        raw_schema = self.chain.invoke(
            {
                "event": event,
                "context": context,
                "examples": self._get_examples(event),
            },
        )

        raw_schema = cast(dict, raw_schema)

        # Error handling for when the output is not parsed correctly
        if not raw_schema.get("parsed"):
            # If the tool has an exception, return it
            if raw_schema.get("parsing_error"):
                return report.failure(raw_schema["parsing_error"])

            # Otherwise try to return the output message from the llm
            if raw_schema.get("out"):
                # The output message should always be an AIMessage,
                # but if it is not, return the raw output.
                try:
                    raw_out = cast(AIMessage, raw_schema["out"])
                    return report.failure(raw_out.text())
                except ValueError:
                    return report.failure(raw_schema["out"])

            return report.failure("The output was not parsed correctly, and no error message was returned by the llm.")

        # Construct the graph from the structured output.
        try:
            output_graph: GraphDocument = raw_schema["parsed"].graph()

            for node in output_graph.nodes:
                node_id = f"http://example.com/lkgb/logs/run/{uuid.uuid4()}"
                node.id = node_id
                node.properties["uri"] = node_id

            return report.success(output_graph)
        except ValueError as e:
            return report.failure(e)
