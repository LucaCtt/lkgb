import logging

import pandas as pd
from langchain.globals import set_debug
from tqdm import tqdm

from lkgb import config
from lkgb.backend import HuggingFaceBackend, OllamaBackend
from lkgb.parser import Parser
from lkgb.reports import RunSummary
from lkgb.store import OntologyStore

set_debug(True)

# Set up logging format
log_formatter = logging.Formatter("%(asctime)s [%(levelname)-4.4s] (%(module)s) %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logging.getLogger().addHandler(console_handler)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set the backend
if config.USE_OLLAMA_BACKEND:
    logger.info("Using Ollama backend")
    backend = OllamaBackend()
else:
    logger.info("Using HuggingFace backend")
    backend = HuggingFaceBackend()

# Load the embeddings model
embeddings = backend.get_embeddings(model=config.EMBEDDINGS_MODEL)

# Create the vector store
ontology = OntologyStore(
    url=config.NEO4J_URL,
    username=config.NEO4J_USERNAME,
    password=config.NEO4J_PASSWORD,
)

# Create the parser model
parser_model = backend.get_parser_model(
    model=config.PARSER_MODEL,
    temperature=config.PARSER_TEMPERATURE,
)

parser = Parser(
    parser_model,
    ontology,
    config.SELF_REFLECTION_STEPS,
)


def main() -> None:
    logger.info("Reading logs from %s", config.TEST_LOG_PATH)

    logs_df = pd.read_csv(config.TEST_LOG_PATH)
    # To prevent weird stuff with NaNs
    logs_df = logs_df.fillna("")

    reports = []
    for log in tqdm(logs_df["text"], desc="Processing logs", colour="blue"):
        report = parser.parse(log)
        reports.append(report)

    logger.info("Log parsing done.")

    summary = RunSummary(reports)

    logger.info("Run summary:")
    logger.info("- Average total time taken to parse each log: %s", summary.avg_total_time_taken())


if __name__ == "__main__":
    main()
