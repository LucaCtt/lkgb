"""Main entry point for the log knowledge graph builder.

Handles the instantiation of the parser and the backend,
the reading of the logs, and the construction of the knowledge graph.
"""

import csv
import logging
from pathlib import Path

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

# Create the vector store
store = EventsStore(
    url=config.neo4j_url,
    username=config.neo4j_username,
    password=config.neo4j_password,
    embeddings=embeddings,
    experiment_id=config.experiment_id,
)

app = typer.Typer()


@app.command()
def clear() -> None:
    store.clear()
    logger.info("Store cleared.")


@app.command()
def parse() -> None:
    logger.info("Experiment ID: %s", config.experiment_id)

    # Load the parser model
    llm = backend.get_parser_model(
        model=config.parser_model,
        temperature=config.parser_temperature,
    )
    logger.info("Language model '%s' loaded.", config.parser_model)

    store.initialize(config.dump())
    logger.info("Store at %s initialized.", config.neo4j_url)

    parser = Parser(llm, store, config.prompt_build_graph, config.self_reflection_steps)
    logger.info("Reading logs from %s", config.test_log_path)

    reports = []

    # Open the test log file once to get the number of lines,
    # without loading it all into memory
    with Path(config.test_log_path).open() as file:
        n_lines = sum(1 for _ in file)

    # Now open the file again and parse the logs
    with Path(config.test_log_path).open() as file:
        events = csv.DictReader(filter(lambda row: row[0] != "#", file))

        for event in track(events, description="Parsing logs", total=n_lines):
            report = parser.parse(event["Log Event"], {"file": event["File"], "device": event["Device"]})
            reports.append(report)

    logger.info("Log parsing done.")

    summary = RunSummary(reports)

    logger.info("Run summary:")
    logger.info("- Average total time taken to parse each log: %s", summary.avg_total_time_taken())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
