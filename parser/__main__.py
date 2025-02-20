import re
import shutil
import uuid
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

from parser import config
from parser.chain import create_chain
from parser.vectore_store import VectorStore

# Load the embeddings model
local_embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDINGS_MODEL, model_kwargs={"trust_remote_code": True})

# Reset the chroma database if the flag is set
if config.RESET_CHROMA_DB and Path.exists(Path(config.CHROMA_PERSIST_DIR)):
    shutil.rmtree(config.CHROMA_PERSIST_DIR)

# Create the vector store
vector_store = VectorStore(config.CHROMA_PERSIST_DIR, local_embeddings)

parser_pipeline = HuggingFacePipeline.from_model_id(
    model_id=config.PARSER_MODEL,
    task="text-generation",
    device_map="auto",
    pipeline_kwargs={"temperature": config.PARSER_TEMPERATURE, "max_length": config.PARSER_NUM_CTX, "truncation": True},
)

# Create the parser model
parser_model = ChatHuggingFace(llm=parser_pipeline)

chain = create_chain(parser_model)


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


def check_all_templates_match(log: str, templates: list[str]) -> bool:
    """
    Checks if a given log string matches all the provided regular expression templates.

    Args:
        log (str): The log string to be checked.
        templates (list[str]): A list of regular expression templates to match against the log.

    Returns:
        bool: True if the log matches all the templates, False otherwise. If there is an error in any of the regular expressions, it returns False.

    """
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
