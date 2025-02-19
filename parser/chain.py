from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough


def format_logs_for_prompt(log: str, similar_logs: list[Document]) -> str:
    """
    Formats the given log and a list of similar logs into a string suitable for use in a prompt.

    This function takes an input log and a list of similar logs, extracts the relevant content from each log,
    and formats them into a single string that can be used as input for a prompt. The logs are enclosed in
    double quotes and separated by commas, and the entire list is enclosed in square brackets.

    Args:
        log (str): The input log to be formatted.
        similar_logs (list[Document]): A list of Document objects representing similar logs.

    Returns:
        str: A formatted string containing the input log and similar logs, suitable for use in a prompt.

    """
    # Cut the "text: " prefix from the page content of each log
    all_logs = [log, *[similar_log.page_content for similar_log in similar_logs]]
    all_logs = [f'"{log}"' for log in all_logs]

    return "[" + ", ".join(all_logs) + "]"


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You will be provided with a list of logs. You must identify and abstract all the dynamic variables in logs with '<*>' and output ONE static log template that matches all the logs. Datetimes and ip addresses should each be abstracted as a standalone '<*>'. Output just the input logs' template, without any other text",
        ),
        (
            "human",
            'Log list: ["2022-01-21 00:09:11 try to connect to host: 172.16.254.1:5000, finished.", "2022-01-21 00:09:11 try to connect to host: 173.16.254.2:6060, finished."]',
        ),
        ("ai", "<*> try to connect to host: <*>, finished."),
        ("human", "Log list: {logs}"),
    ],
)


def create_chain(parser_model: Runnable) -> Runnable:
    return (
        RunnablePassthrough.assign(
            logs=lambda inputs: format_logs_for_prompt(inputs["input_log"], inputs["similar_logs"]),
        )
        | prompt
        | parser_model
        | StrOutputParser()
    )
