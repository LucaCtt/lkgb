import re
import uuid

import config
from langchain_core.documents import Document
from langchain_core.globals import set_debug, set_verbose
from langchain_ollama import ChatOllama, OllamaEmbeddings

from parser.chain import create_chain
from parser.vectore_store import create_vector_store

set_verbose(False)
set_debug(False)

# Load the embeddings model
local_embeddings = OllamaEmbeddings(model=config.EMBEDDINGS_MODEL)

# Create the vector store
vector_store = create_vector_store(config.TEST_LOG_PATH, local_embeddings)

# Create the parser model
parser_model = ChatOllama(model=config.PARSER_MODEL, temperature=config.PARSER_TEMPERATURE)

chain = create_chain(parser_model)


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
    similarity_question = f'Which logs are most similar to "{log}"?'

    # Check if there are very similar logs
    # Assumption: the returned documents are sorted by most relevant first
    very_similar_logs = vector_store.similarity_search_with_relevance_scores(
        similarity_question,
        score_threshold=0.7,
        k=10,
        filter={"template": {"$ne": ""}},
    )

    # If there are very similar logs,
    # check if their template matches with the current log
    if len(very_similar_logs) > 0:
        for similar_log in very_similar_logs:
            if re.match(similar_log[0].metadata["template"], log):
                return similar_log[0].metadata["template"]

    # If there are no very similar logs or their template doesn't match,
    # find sufficiently similar logs
    similar_logs = vector_store.similarity_search_with_relevance_scores(similarity_question, k=5, score_threshold=0.5)
    similar_logs = [log[0] for log in similar_logs]

    # Perform self-reflection to verify that the template
    # matches both the current and similar logs
    self_reflection_countdown = config.SELF_REFLECTION_STEPS

    while self_reflection_countdown > 0:
        self_reflection_countdown -= 1

        # Find the template using the current log and the similar logs
        template = chain.invoke({"input_log": log, "similar_logs": similar_logs})

        # Replace all of the <*> in the template with (.*?)
        template = template.replace("<*>", "(.*?)")

        # Check that the current log matches the template
        if not re.match(template, log):
            continue

        # Check that all the similar logs match the template
        for similar_log in similar_logs:
            if not re.match(template, similar_log.page_content):
                continue

        # If the template matches all the logs, stop the self-reflection loop
        break

    # Update the template metadata value for the similar logs
    for similar_log in similar_logs:
        similar_log.metadata["template"] = template
        vector_store.update_document(document_id=similar_log.id, document=similar_log)

    # Save the new logs to the vector store
    vector_store.add_documents([Document(id=uuid.uuid4(), page_content=log, metadata={"template": template})])

    return template


if __name__ == "__main__":
    get_template("2022-01-21 01:04:19 jhall/192.168.230.165:46011 peer info: IV_TCPNL=1")
