import re
from typing import cast

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from pydantic import BaseModel, Field

from lkgb.ontology import Ontology
from lkgb.reports import ParserReport
from lkgb.store import Store


class _LogEvent(BaseModel):
    template: str = Field(
        description="The template of the log event, where identified nodes are replaced with placeholders.",
    )


def _create_graph_structure(
    classes: list[tuple[str, str]],
) -> type[_LogEvent]:
    """Create a graph structure from a list of ontology classes.

    Args:
        classes (list[tuple[str, str]]): A list of ontology classes where each tuple
        contains the class name and its description.

    Returns:
        _GraphStructure: A graph structure containing the nodes of the ontology classes.

    """

    class DynamicLogEvent(_LogEvent):
        template: str = Field(
            description=f"""The template of the log event, where identified nodes are replaced with class placeholders.
            Available placeholders with their descriptions are: {[f"{cls[0]}: {cls[1]}" for cls in classes]}""",
        )

    return DynamicLogEvent


system_prompt = (
    "## 1. Overview\n"
    "You are a top-tier algorithm designed for extracting templates with structured node information from log events.\n"
    "Try to capture as much information from the log event as possible without sacrificing accuracy."
    "Do not add any information that is not explicitly mentioned in the log.\n"
    "- **Template**: the full log event where identified nodes are replaced with labels between double angled braces.\n"
    "- **Nodes** represent entities and concepts of the log event.\n"
    "## 2. Labeling Nodes\n"
    "- **Consistency**: Ensure you use available types for nodes and relationship labels, respecting their description."
    "Ensure you use basic or elementary types for labels."
    "For example, when you identify a node representing an User, always label it as **'User'**."
    "Avoid using more specific terms like 'RootUser' or 'Bob'.\n"
    "## 4. Strict Compliance\n"
    "Adhere to the rules strictly. Non-compliance will result in termination."
)

gen_template_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        ("placeholder", "{examples}"),
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


def _check_template_match(log: str, template: str) -> bool:
    """Check if a given log string matches a specified template using regular expressions.

    Args:
        log (str): The log string to be checked.
        template (str): The regular expression template to match against the log string.

    Returns:
        bool: True if the log matches the template, False otherwise. If the template is invalid, returns False.

    """
    try:
        regex = re.sub(r"<<.*?>>", r"(.*?)", template)
        return re.match(regex, log) is not None
    except re.error:
        return False


class Parser:
    """The Parser class is responsible for parsing log events and identifying their templates."""

    def __init__(
        self,
        parser_model: BaseLanguageModel,
        store: Store,
        ontology: Ontology,
        self_reflection_steps: int,
    ) -> "Parser":
        self.history = ChatMessageHistory()
        self.store = store
        self.self_reflection_steps = self_reflection_steps

        try:
            parser_model.with_structured_output(_LogEvent)
        except NotImplementedError as e:
            msg = "The parser model must support structured output."
            raise ValueError(msg) from e

        structured_model = parser_model.with_structured_output(
            _create_graph_structure(ontology.classes()),
            include_raw=True,
        )

        chain = gen_template_prompt | structured_model
        self.chain = RunnableWithMessageHistory(
            chain,
            lambda _: self.history,
            input_messages_key="log",
            history_messages_key="messages",
        )

        self.ontology = ontology

    def __build_graph(self, log: str, template: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        regex = re.sub(r"<<(.*?)>>", r"(?P<\1>.*?)", template)
        match = re.match(regex, log)

        nodes = [(key, match.group(key)) for key in match.groupdict()]

        relationships = []
        for node in nodes:
            rel = self.ontology.get_event_object_property(node[0])
            if rel:
                relationships.append((rel, node[0]))

        return nodes, relationships

    def parse(self, log: str) -> ParserReport:
        """Given a log, this function identifies and updates the template for the log and also for similar logs.

        It first searches for very similar logs in the store and checks if their templates match the current log.
        If no matching template is found, it searches for sufficiently similar logs
        and uses them to generate a template.

        Args:
            log (str): The log for which the template needs to be identified.

        Returns:
            ParserReport: A ParserReport object containing the timings of the parsing operations.

        """
        report = ParserReport()

        # Check if there are very similar logs
        # Assumption: the returned documents are sorted by most relevant first
        very_similar_logs = self.store.find_similar_events_with_template(log, 0.7)

        report.find_very_similar_logs_done()

        # If there are very similar logs,
        # check if their template matches with the current log
        for similar_log in very_similar_logs:
            template = similar_log.metadata["template"]
            if _check_template_match(log, template):
                self.store.add_event(log, template)
                return report.finish()

        # If there are no very similar logs or their template doesn't match,
        # find sufficiently similar logs for RAG examples
        similar_logs = self.store.find_similar_events_with_template(log, 0.5)

        report.find_similar_logs_done()

        # Perform self-reflection to verify that the template
        # matches both the current and similar logs
        self_reflection_countdown = self.self_reflection_steps

        while self_reflection_countdown > 0:
            self_reflection_countdown -= 1

            # Find the template using the current log and the similar logs
            raw_schema = self.chain.invoke(
                {"log": log, "examples": _get_examples(similar_logs)},
                {"configurable": {"session_id": "unused"}},
            )

            raw_schema = cast(dict, raw_schema)
            template = ""

            if not raw_schema["parsed"] or not _check_template_match(log, raw_schema["parsed"].template):
                if self_reflection_countdown > 0:
                    self.history.add_user_message(
                        "The template you generated is invalid. Please try again.",
                    )
                continue

            template = raw_schema["parsed"].template

            # If the template matches the log, stop the self-reflection loop
            break

        report.template_generation_done()

        # Save the new logs to the store
        self.store.add_event(log, template)

        return report.finish()
