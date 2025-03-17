from typing import Self, cast

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from pydantic import BaseModel, Field, model_validator

from lkgb.reports import ParserReport
from lkgb.store import OntologyStore
from lkgb.tools import fetch_ip_address_info


class _LogGraph(BaseModel):
    triples: list = Field(description="The RDF graph representing the log event.")


system_prompt = (
    "## 1. Overview\n"
    "You are a top-tier algorithm designed for extracting information"
    "in structured formats to build a knowledge graph according to an ontology.\n"
    "Try to capture as much information from the text as possible without sacrificing accuracy."
    "Do not add any information that is not explicitly mentioned in the text.\n"
    "- **Triples** represent connections between entities or concepts."
    "They consist of a subject, predicate, and object.\n"
    "- The aim is to achieve exhaustiveness in the knowledge graph, making it ontology-compliant.\n"
    "## 2. Labeling Triples\n"
    "- The labels for subjects, predicates, and objects should be consistent with the ontology."
    "- **IDs**: IDs for subjects and objects should be unique and consistent."
    "- **Consistency**: Ensure you use available types for triples labels."
    "Ensure you use the most specific types for triples labels.\n"
    "Ensure consistency and generality in relationship types when constructing knowledge graphs"
    "## 3. Strict Compliance\n"
    "Adhere to the rules strictly. Non-compliance will result in termination."
)

gen_graph_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        ("human", "Log: '{log}'"),
        ("placeholder", "{messages}"),
    ],
)


def _get_examples(similar_logs: list[Document]) -> list[BaseMessage]:
    def get_example_group(log: str, template: str) -> list[BaseMessage]:
        return [
            HumanMessage(f"Log: '{log}", name="example_user"),
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
            parser_model.with_structured_output(_LogGraph)
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

    def __create_graph_structure(self) -> type[_LogGraph]:
        valid_triples = self.ontology.triples()

        valid_subjects = {subj for subj, _, _ in valid_triples}
        valid_predicates = {pred for _, pred, _ in valid_triples}
        valid_objects = {obj for _, _, obj in valid_triples}

        class _Triple(BaseModel):
            subject_id: str = Field(description="The ID of the subject.")
            subject_label: str = Field(
                description=f"The ontology label of the subject, available labels are: {valid_subjects}",
                enum=valid_subjects,
            )
            subject_value: str = Field(description="The value of the subject.")

            predicate_label: str = Field(
                description=f"The ontology label of the predicate, available labels are: {valid_predicates}",
                enum=valid_predicates,
            )

            object_id: str = Field(description="The ID of the object.")
            object_label: str = Field(
                description=f"The ontology label of the object, available labels are: {valid_objects}",
                enum=valid_objects,
            )
            object_value: str = Field(description="The value of the object.")

            @model_validator(mode="after")
            def check_valid_combination(self) -> Self:
                if (self.subject_label, self.predicate_label, self.object_label) not in valid_triples:
                    msg = (
                        "Invalid triple combination: ",
                        f"({self.subject_label}, {self.predicate_label}, {self.object_label})",
                    )
                    raise ValueError(msg)
                return self

        class DynamicLogGraph(_LogGraph):
            triples: list[_Triple] = Field(
                description="The RDF graph representing the log event. \
                    Each triple consists of a subject, predicate, and object. \
                    The predicate domain and range are defined in the ontology.",
            )

        return DynamicLogGraph

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
        triples = raw_schema["parsed"].triples

        report.template_generation_done()

        print(triples)  # noqa: T201
        return report.finish()
