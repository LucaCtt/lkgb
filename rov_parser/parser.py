import re

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableWithMessageHistory

from rov_parser.vector_store import VectorStore

# For deepseek R1 it's recommended to input all instructions in a user prompt.
gen_template_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """I will provide you with a list of logs. You must identify and abstract all the dynamic variables in logs with '<*>' and output ONE static log template that matches all the logs. You are also provided with a list of rules to follow and an example to understand the task.

                [Rules]
                    1. Datetimes and ip addresses should each be abstracted as a standalone '<*>'.
                    2. Please reason step by step, and put your final answer within '\\boxed{{}}'.
                    3. Initiate your response with \"<think>\\n\" at the beginning of every output.

                [Example]
                ["2022-01-21 00:09:11 try to connect to host: 172.16.254.1:5000, finished.", "2022-01-21 00:09:11 try to connect to host: 173.16.254.2:6060, finished."] -> <*> try to connect to host: <*>, finished.

                [Input]
                {logs}
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ],
)

correct_template_prompt = ChatPromptTemplate.from_messages(
    [("human", 'The template you generated is invalid for log "{log}". Please try again.')],
)


def format_logs_for_prompt(logs: list[str]) -> str:
    """
    Formats the given logs into a string suitable for use in a prompt.

    Args:
        logs (list[str]): A list of logs.

    Returns:
        str: A formatted string containing the input logs, suitable for use in a prompt.

    """
    # Cut the "text: " prefix from the page content of each log
    all_logs = [f'"{log}"' for log in logs]

    return "[" + ", ".join(all_logs) + "]"


def template_to_regex(template: str) -> str:
    """
    Converts a template string with placeholders into a regular expression string.

    The function replaces the placeholder "<*>" in the template with the regex pattern "(.*?)",
    which matches any character sequence. It also corrects small errors the parser might make.

    Args:
        template (str): The template string containing placeholders.

    Returns:
        str: The resulting regular expression string.

    """
    # Extract the string within \boxed{} from the template
    match = re.search(r"\\boxed{(.*?)}", template)
    if match:
        template = match.group(1)

    # Replace <*> with the regex pattern (.*?)
    regex = template.replace("<*>", "(.*?)").strip()

    # Remove any quotes around the regex
    regex = regex.removeprefix("'").removesuffix("'")

    # Remove any extra spaces from the regex
    return regex.strip()


def check_template_match(log: str, template: str) -> bool:
    """
    Checks if a given log string matches a specified template using regular expressions.

    Args:
        log (str): The log string to be checked.
        template (str): The regular expression template to match against the log string.

    Returns:
        bool: True if the log matches the template, False otherwise. If the template is invalid, returns False.

    """
    try:
        return re.match(template, log) is not None
    except re.error:
        return False


class Parser:
    def __init__(
        self,
        parser_model: Runnable,
        vector_store: VectorStore,
        memory_match_min_quality: int,
        self_reflection_steps: int,
    ) -> "Parser":
        self.history = ChatMessageHistory()
        self.vector_store = vector_store
        self.memory_match_min_quality = memory_match_min_quality
        self.self_reflection_steps = self_reflection_steps

        chain = (
            RunnablePassthrough.assign(logs=lambda inputs: format_logs_for_prompt(inputs["logs"]))
            | gen_template_prompt
            | parser_model
            | StrOutputParser()
        )
        self.chain = RunnableWithMessageHistory(
            chain,
            lambda _: self.history,
            input_messages_key="logs",
            history_messages_key="messages",
        )

    def compute_template(self, log: str) -> None:
        """
        Given a log, this function identifies and updates the template for the log and also for similar logs.
        It first searches for very similar logs in the vector store and checks if their templates match the current log.
        If no matching template is found, it searches for sufficiently similar logs and uses them to generate a template.

        Args:
            log (str): The log for which the template needs to be identified.

        """
        # Check if there are very similar logs
        # Assumption: the returned documents are sorted by most relevant first
        very_similar_logs = self.vector_store.find_very_similar_logs_with_template(log)

        # If there are very similar logs,
        # check if their template matches with the current log
        if len(very_similar_logs) >= self.memory_match_min_quality and self.__check_all_templates_match(
            log,
            [similar.metadata["template"] for similar in very_similar_logs],
        ):
            self.vector_store.add_document(log, very_similar_logs[0].metadata["template"])
            return

        # If there are no very similar logs or their template doesn't match,
        # find sufficiently similar logs
        similar_logs = self.vector_store.find_similar_logs(log)

        all_logs = [log, *[similar_log.page_content[len("search_document: ") :] for similar_log in similar_logs]]

        # Perform self-reflection to verify that the template
        # matches both the current and similar logs
        self_reflection_countdown = self.self_reflection_steps

        while self_reflection_countdown > 0:
            self_reflection_countdown -= 1

            # Find the template using the current log and the similar logs
            template = self.chain.invoke(
                {"logs": all_logs},
                {"configurable": {"session_id": "unused"}},
            )

            template_regex = template_to_regex(template)

            # Check that all logs match the template
            for current_log in all_logs:
                if not check_template_match(current_log, template_regex):
                    template_regex = None
                    self.history.add_user_message(
                        f'The template you generated is invalid for log "{current_log}". Please try again.',
                    )
                    continue

            # If the template matches all the logs, stop the self-reflection loop
            break

        # Update the template metadata value for the similar logs
        for similar_log in similar_logs:
            if similar_log.metadata["template"] == template_regex:
                continue

            similar_log.metadata["template"] = template_regex
            self.vector_store.update_document(similar_log)

        # Save the new logs to the vector store
        self.vector_store.add_document(log, template_regex)
