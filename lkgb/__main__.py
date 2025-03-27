"""Main entry point for the log knowledge graph builder.

Handles the instantiation of the parser and the backend,
the reading of the logs, and the construction of the knowledge graph.
"""

import logging

import typer
from rich.logging import RichHandler
from rich.progress import track

from lkgb.accuracy import graph_edit_distance
from lkgb.backend import HuggingFaceBackend, OllamaBackend
from lkgb.config import Config
from lkgb.parser import Parser, RunSummary
from lkgb.store import Store

config = Config()

# Set up logging format
logging.basicConfig(format="%(message)s", handlers=[RichHandler(omit_repeated_times=False)])
logger = logging.getLogger("rich")
logger.setLevel(logging.INFO)

# Set the backend
backend = OllamaBackend() if config.use_ollama_backend else HuggingFaceBackend()

# Load the embeddings model
embeddings = backend.get_embeddings(model=config.embeddings_model)

# Create the vector store
store = Store(config=config, embeddings=embeddings)

app = typer.Typer()


@app.command()
def clear() -> None:
    store.clear()
    logger.info("Store cleared.")


@app.command()
def parse() -> None:
    logger.info("Using %s backend.", "Ollama" if config.use_ollama_backend else "HuggingFace")
    logger.info("Experiment ID: %s", config.experiment_id)
    logger.info("Embeddings model: '%s'", config.embeddings_model)

    # Load the parser model
    llm = backend.get_parser_model(
        model=config.parser_model,
        temperature=config.parser_temperature,
    )
    logger.info("Language model: '%s'", config.parser_model)

    store.initialize()
    logger.info("Store at '%s' initialized.", config.neo4j_url)

    test_events = store.dataset.tests()
    logger.info("Read %d tests from '%s'", len(test_events), config.tests_path)

    parser = Parser(llm, store, config.prompt_build_graph, config.self_reflection_steps)

    reports = []
    average_ged = 0
    for event, context, graph in track(test_events, description="Parsing events"):
        report = parser.parse(event, context)
        reports.append(report)

        if report.error is not None:
            logger.warning("Event could not be parsed: %s", report.error)
        elif report.graph is not None:
            store.dataset.add_event_graph(report.graph)
            average_ged += graph_edit_distance(report.graph, graph)
            logger.debug("GED: %f", graph_edit_distance(report.graph, graph))
        else:
            logger.warning("Event was parsed but no graph was generated.")

    average_ged /= len(test_events)

    logger.info("Log parsing done.")

    summary = RunSummary(reports)

    logger.info("Run summary:")
    logger.info("- Average parse time per event: %f seconds", summary.parse_time_average())
    logger.info("- Success percentage: %f%%", summary.success_percentage() * 100)
    logger.info("- Average GED: %f", average_ged)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
