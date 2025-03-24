"""Main entry point for the log knowledge graph builder.

Handles the instantiation of the parser and the backend,
the reading of the logs, and the construction of the knowledge graph.
"""

import logging

import pandas as pd
import typer
from rich.logging import RichHandler
from rich.progress import track

from lkgb.backend import HuggingFaceBackend, OllamaBackend
from lkgb.config import Config
from lkgb.parser import Parser
from lkgb.reports import RunSummary
from lkgb.store import EventsStore

config = Config()

# Set up logging format
logging.basicConfig(format="%(message)s", handlers=[RichHandler(omit_repeated_times=False)])

logger = logging.getLogger("rich")
logger.setLevel(logging.INFO)

logger.info("Experiment ID: %s", config.experiment_id)

# Set the backend
if config.use_ollama_backend:
    logger.info("Using Ollama backend.")
    backend = OllamaBackend()
else:
    logger.info("Using HuggingFace backend.")
    backend = HuggingFaceBackend()

# Load the embeddings model
embeddings = backend.get_embeddings(model=config.embeddings_model)
logger.info("Embeddings model '%s' loaded.", config.embeddings_model)

# Load the parser model
llm = backend.get_parser_model(
    model=config.parser_model,
    temperature=config.parser_temperature,
)
logger.info("Language model '%s' loaded.", config.parser_model)

# Create the vector store
store = EventsStore(
    url=config.neo4j_url,
    username=config.neo4j_username,
    password=config.neo4j_password,
    embeddings=embeddings,
    experiment_id=config.experiment_id,
)
store.initialize(config.dump())
logger.info("Store at %s initialized.", config.neo4j_url)

parser = Parser(llm, store, config.prompt_build_graph, config.self_reflection_steps)

app = typer.Typer()


@app.command()
def clean() -> None:
    logger.info("Cleaning up the store.")
    store.clean()
    logger.info("Store cleaned.")


@app.command()
def parse() -> None:
    logger.info("Reading logs from %s", config.test_log_path)

    events_df = pd.read_csv(config.test_log_path, comment="#")
    # To prevent weird stuff with NaNs
    events_df = events_df.fillna("")

    reports = []
    for event in track(events_df["Log Event"], description="Processing logs"):
        report = parser.parse(event)
        reports.append(report)

    logger.info("Log parsing done.")

    summary = RunSummary(reports)

    logger.info("Run summary:")
    logger.info("- Average total time taken to parse each log: %s", summary.avg_total_time_taken())


if __name__ == "__main__":
    app()
