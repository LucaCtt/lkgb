"""Main entry point for the log knowledge graph builder.

Handles the instantiation of the parser and the backend,
the reading of the logs, and the construction of the knowledge graph.
"""

import logging

import pandas as pd
from tqdm import tqdm

from lkgb.backend import HuggingFaceBackend, OllamaBackend
from lkgb.config import Config
from lkgb.parser import Parser
from lkgb.reports import RunSummary
from lkgb.store import EventsStore

# Set up logging format
log_formatter = logging.Formatter("%(asctime)s [%(levelname)-4.4s] (%(module)s) %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logging.getLogger().addHandler(console_handler)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set the backend
if Config.use_ollama_backend:
    logger.info("Using Ollama backend")
    backend = OllamaBackend()
else:
    logger.info("Using HuggingFace backend")
    backend = HuggingFaceBackend()

# Load the embeddings model
embeddings = backend.get_embeddings(model=Config.embeddings_model)

# Create the vector store
store = EventsStore(
    url=Config.neo4j_url,
    username=Config.neo4j_username,
    password=Config.neo4j_password,
    embeddings=embeddings,
    experiment_id=Config.experiment_id,
)

store.clear()
store.initialize(Config.ontology_path, Config.examples_path)

# Create the parser model
parser_model = backend.get_parser_model(
    model=Config.parser_model,
    temperature=Config.parser_temperature,
)

parser = Parser(
    parser_model,
    store,
    Config.self_reflection_steps,
)


def main() -> None:
    logger.info("Reading logs from %s", Config.test_log_path)

    events_df = pd.read_csv(Config.test_log_path, comment="#")
    # To prevent weird stuff with NaNs
    events_df = events_df.fillna("")

    reports = []
    for event in tqdm(events_df["Log Event"], desc="Processing logs", colour="blue"):
        report = parser.parse(event)
        reports.append(report)

    logger.info("Log parsing done.")

    summary = RunSummary(reports)

    logger.info("Run summary:")
    logger.info("- Average total time taken to parse each log: %s", summary.avg_total_time_taken())


if __name__ == "__main__":
    main()
