import re
import uuid

from langchain_core.documents import Document
from langchain_core.globals import set_debug, set_verbose
from langchain_ollama import ChatOllama, OllamaEmbeddings

from parser import config
from parser.chain import create_chain
from parser.vectore_store import VectorStore

set_verbose(False)
set_debug(False)

# Load the embeddings model
local_embeddings = OllamaEmbeddings(model=config.EMBEDDINGS_MODEL)

# Create the vector store
vector_store = VectorStore(config.CHROMA_PERSIST_DIR, local_embeddings)

# Create the parser model
parser_model = ChatOllama(model=config.PARSER_MODEL, temperature=config.PARSER_TEMPERATURE)

chain = create_chain(parser_model)


def template_to_regex(template: str) -> str:
    regex = template.replace("<*>", "(.*?)").strip()

    # Remove the quotes from the regex if they exist
    regex = regex.removeprefix("'").removesuffix("'")

    # Remove any extra spaces from the regex
    return regex.strip()


def get_template(log: str) -> str:
    """
    Given a log, this function identifies and returns a template that matches the log.
    It first searches for very similar logs in the vector store and checks if their templates match the current log.
    If no matching template is found, it searches for sufficiently similar logs and uses them to generate a template.

    Args:
        log (str): The log for which the template needs to be identified.

    Returns:
        str: The identified template for the given log.

    """
    # Check if there are very similar logs
    # Assumption: the returned documents are sorted by most relevant first
    very_similar_logs = vector_store.find_very_similar_logs_with_template(log)

    # If there are very similar logs,
    # check if their template matches with the current log
    if len(very_similar_logs) > 0:
        for similar_log in very_similar_logs:
            if re.match(similar_log.metadata["template"], log):
                return similar_log.metadata["template"]

    # If there are no very similar logs or their template doesn't match,
    # find sufficiently similar logs
    similar_logs = vector_store.find_similar_logs(log)

    # Perform self-reflection to verify that the template
    # matches both the current and similar logs
    self_reflection_countdown = config.SELF_REFLECTION_STEPS

    while self_reflection_countdown > 0:
        self_reflection_countdown -= 1

        # Find the template using the current log and the similar logs
        template = chain.invoke({"input_log": log, "similar_logs": similar_logs})

        template_regex = template_to_regex(template)

        # Check that the current log matches the template
        if not re.match(template_regex, log):
            continue

        # Check that all the similar logs match the template
        for similar_log in similar_logs:
            if not re.match(template_regex, similar_log.page_content):
                continue

        # If the template matches all the logs, stop the self-reflection loop
        break

    # Update the template metadata value for the similar logs
    for similar_log in similar_logs:
        similar_log.metadata["template"] = template_regex
        vector_store.update_document(similar_log)

    # Save the new logs to the vector store
    vector_store.add_document(Document(id=uuid.uuid4(), page_content=log, metadata={"template": template_regex}))

    return template_regex


if __name__ == "__main__":
    template = get_template("2022-01-21 01:04:19 jhall/192.168.230.165:46011 peer info: IV_TCPNL=1")

    print(template)
