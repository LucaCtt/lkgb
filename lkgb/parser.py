from typing import cast

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from pydantic import BaseModel, Field

from lkgb.reports import ParserReport
from lkgb.store import OntologyStore
from lkgb.tools import fetch_ip_address_info


class _EventGraph(BaseModel):
    triples: list = Field(description="The triples representing the event.")


system_prompt = (
    "## 1. Overview\n"
    "You are a top-tier algorithm designed for extracting information from a log event "
    "in structured formats to build a knowledge graph according to an ontology.\n"
    "Try to capture as much information from the event as possible without sacrificing accuracy."
    "Only add information that is defined in the ontology.\n"
    "Do not add any information that is not explicitly mentioned in the event.\n"
    "## 2. Knowledge Graph\n"
    "- **Triples** represent connections between entities or concepts."
    "They consist of a subject, predicate, and object. The subject and object have an id, a label, and a value."
    "The predicate has just a label.\n"
    "- The aim is to achieve exhaustiveness in the knowledge graph, making it ontology-compliant.\n"
    "## 3. Labels\n"
    "- **Consistency**: The labels for subjects, predicates, and objects must be consistent with the ontology. "
    "Ensure you use available types for the labels."
    "- **IDs**: IDs for subjects and objects must be unique and consistent."
    "## 4. Strict Compliance\n"
    "Adhere to the rules strictly. Non-compliance will result in termination."
)

gen_graph_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        ("human", "Event: '{log}'"),
        ("placeholder", "{messages}"),
    ],
)


def _get_examples(similar_logs: list[Document]) -> list[BaseMessage]:
    def get_example_group(log: str, template: str) -> list[BaseMessage]:
        return [
            HumanMessage(f"Event: '{log}", name="example_user"),
            AIMessage(
                "",
                name="example_assistant",
                tool_calls=[
                    {
                        "name": "DynamicLogEvent",
                        "args": {
                            "template": template,
                        },
                        "id": "1",
                    },
                ],
            ),
            ToolMessage("", tool_call_id="1"),
        ]

    if not similar_logs:
        return get_example_group(
            "2022-01-21 00:09:11 jhall/192.168.230.165:46011 VERIFY EKU OK",
            "2022-01-21 00:09:11 <<User>>/<<Address>> VERIFY EKU OK",
        )

    message_groups = []
    for similar_log in similar_logs:
        message_groups.extend(
            get_example_group(similar_log.page_content[len("search_document: ") :], similar_log.metadata["template"]),
        )
    return message_groups


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

        structured_model = parser_model.bind_tools([fetch_ip_address_info])
        structured_model = structured_model.with_structured_output(self.__create_graph_structure())

        chain = gen_graph_prompt | structured_model
        self.chain = RunnableWithMessageHistory(
            chain,
            lambda _: self.history,
            input_messages_key="log",
            history_messages_key="messages",
        )

    def __create_graph_structure(self) -> type[_EventGraph]:
        valid_triples = self.ontology.triples()

        valid_subjects = {subj for subj, _, _ in valid_triples}
        valid_predicates = {pred for _, pred, _ in valid_triples}
        valid_objects = {obj for _, _, obj in valid_triples}

        class _Subject(BaseModel):
            id: str = Field(description="The ID of the subject.")
            label: str = Field(
                description=f"The ontology label of the subject, available labels are: {valid_subjects}",
                enum=valid_subjects,
            )
            value: str | None = Field(description="The value of the subject.", default=None)

        class _Predicate(BaseModel):
            label: str = Field(
                description=f"The ontology label of the predicate, available labels are: {valid_predicates}",
                enum=valid_predicates,
            )

        class _Object(BaseModel):
            id: str = Field(description="The ID of the object.")
            label: str = Field(
                description=f"The ontology label of the object, available labels are: {valid_objects}",
                enum=valid_objects,
            )
            value: str | None = Field(description="The value of the object.", default=None)

        class _Triple(BaseModel):
            subject: _Subject = Field(description="The subject of the triple.")
            predicate: _Predicate = Field(description="The predicate of the triple.")
            object: _Object = Field(description="The object of the triple.")

        class DynamicEventGraph(_EventGraph):
            triples: list[_Triple] = Field(description="The triples representing the event.")

        DynamicEventGraph.__doc__ = (
            "Your task is to extract triples from text strictly adhering "
            "to the provided schema. The relationships can only appear "
            "between specific node types are presented in the schema format "
            "like: (subject label, relationship label, entity label) /n"
            f"Provided schema is {[f'({subj},{pred},{obj})' for subj, pred, obj in valid_triples]}"
        )

        return DynamicEventGraph

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

        # Find the template using the current log and the similar logs
        raw_schema = self.chain.invoke(
            {"log": log},
            {"configurable": {"session_id": "unused"}},
        )

        raw_schema = cast(dict, raw_schema)
        triples = raw_schema.triples

        return report.finish()
