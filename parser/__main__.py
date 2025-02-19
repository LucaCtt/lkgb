import re
import shutil
import uuid
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_core.globals import set_debug, set_verbose
from langchain_ollama import ChatOllama, OllamaEmbeddings
from tqdm import tqdm

from parser import config
from parser.chain import create_chain
from parser.vectore_store import VectorStore

set_verbose(False)
set_debug(False)

# Load the embeddings model
local_embeddings = OllamaEmbeddings(model=config.EMBEDDINGS_MODEL)


if config.RESET_CHROMA_DB and Path.exists(Path(config.CHROMA_PERSIST_DIR)):
    shutil.rmtree(config.CHROMA_PERSIST_DIR)

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


def check_template_match(log: str, template: str) -> bool:
    try:
        return re.match(template, log) is not None
    except re.error:
        return False


def check_all_templates_match(log: str, templates: list[str]) -> bool:
    try:
        return all(re.match(template, log) for template in templates)
    except re.error:
        return False


def compute_template(log: str) -> None:
    """
    Given a log, this function identifies and updates the template for the log and also for similar logs.
    It first searches for very similar logs in the vector store and checks if their templates match the current log.
    If no matching template is found, it searches for sufficiently similar logs and uses them to generate a template.

    Args:
        log (str): The log for which the template needs to be identified.

    """
    # Check if there are very similar logs
    # Assumption: the returned documents are sorted by most relevant first
    very_similar_logs = vector_store.find_very_similar_logs_with_template(log)

    # If there are very similar logs,
    # check if their template matches with the current log
    if len(very_similar_logs) >= config.MEMORY_MATCH_MIN_QUALITY and check_all_templates_match(
        log,
        [similar.metadata["template"] for similar in very_similar_logs],
    ):
        vector_store.add_document(
            Document(
                id=uuid.uuid4(),
                page_content=log,
                metadata={"template": very_similar_logs[0].metadata["template"]},
            ),
        )
        return

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
        if not check_template_match(log, template_regex):
            continue

        # Check that all the similar logs match the template
        for similar_log in similar_logs:
            if not check_template_match(similar_log.page_content, template_regex):
                continue

        # If the template matches all the logs, stop the self-reflection loop
        break

    # Update the template metadata value for the similar logs
    for similar_log in similar_logs:
        if similar_log.metadata["template"] == template_regex:
            continue

        similar_log.metadata["template"] = template_regex
        vector_store.update_document(similar_log)

    # Save the new logs to the vector store
    vector_store.add_document(Document(id=uuid.uuid4(), page_content=log, metadata={"template": template_regex}))


if __name__ == "__main__":
    logs_df = pd.read_csv(config.TEST_LOG_PATH)
    logs_df = logs_df.fillna("")

    for log in tqdm(logs_df["text"], desc="Processing logs"):
        compute_template(log)

    with Path.open(config.TEST_OUT_PATH, "w") as out_file:
        out_file.write("text,template\n")
        for log in tqdm(logs_df["text"], desc="Writing output"):
            template = vector_store.get_template(log)
            out_file.write(f"{log},{template}\n")
