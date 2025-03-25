"""Main entry point for the log knowledge graph builder.

Handles the instantiation of the parser and the backend,
the reading of the logs, and the construction of the knowledge graph.
"""

import logging

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

    test_events = store.get_tests()
    logger.info("Read %d tests from '%s'", len(test_events), config.tests_path)

    parser = Parser(llm, store, config.prompt_build_graph, config.self_reflection_steps)

    reports = []
    for test in track(test_events, description="Parsing events"):
        report = parser.parse(test.event, test.context)
        reports.append(report)

        if report.error:
            logger.error("Error parsing log: %s", report.error)
        else:
            store.add_event_graph(report.graph)

    logger.info("Log parsing done.")

    summary = RunSummary(reports)

    logger.info("Run summary:")
    logger.info("- Average log parse time: %s", summary.parse_time_average())
    logger.info("- Success percentage: %s", summary.success_percentage())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
