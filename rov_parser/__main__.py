import logging
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from rov_parser import config
from rov_parser.backend import HuggingFaceBackend, OllamaBackend
from rov_parser.parser import Parser
from rov_parser.reports import RunSummary
from rov_parser.vector_store import VectorStore

# Create the output directories if they don't exist
if not Path.exists(Path(config.LOGS_OUT_DIR)):
    Path.mkdir(Path(config.LOGS_OUT_DIR))

# Reset the chroma database if the flag is set
if config.RESET_CHROMA_DB and Path.exists(Path(config.CHROMA_PERSIST_DIR)):
    shutil.rmtree(config.CHROMA_PERSIST_DIR)

# Disable Chroma info logging
logging.getLogger("langchain_core").setLevel(logging.ERROR)

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
local_embeddings = backend.get_embeddings(model=config.EMBEDDINGS_MODEL)

# Create the vector store
vector_store = VectorStore(config.CHROMA_PERSIST_DIR, local_embeddings)

# Create the parser model
parser_model = backend.get_parser_model(
    model=config.PARSER_MODEL,
    temperature=config.PARSER_TEMPERATURE,
)

parser = Parser(parser_model, vector_store, config.MEMORY_MATCH_MIN_QUALITY, config.SELF_REFLECTION_STEPS)

if __name__ == "__main__":
    logger.info("Reading logs from %s", config.TEST_LOG_PATH)

    logs_df = pd.read_csv(config.TEST_LOG_PATH)
    # This is necessary because the Chroma CSV reader doesn't handle NaNs
    logs_df = logs_df.fillna("")

    reports = []
    for log in tqdm(logs_df["text"], desc="Processing logs", colour="blue"):
        report = parser.parse(log)
        reports.append(report)

    logger.info("Log parsing done.")

    with Path.open(config.TEST_OUT_PATH, "w") as out_file:
        out_file.write("text,template\n")
        for log in tqdm(logs_df["text"], desc=f"Writing outputs to {config.TEST_OUT_PATH}", colour="blue"):
            template = vector_store.get_template(log)
            out_file.write(f"{log},{template}\n")

    logger.info("Output written.")

    summary = RunSummary(reports)

    logger.info("Run summary:")
    logger.info("- Logs parsed: %s", len(logs_df))
    logger.info("- Average time taken to find very similar logs: %s", summary.avg_very_similar_logs_time_taken())
    logger.info("- Percentage of memory matches: %s", summary.percentage_memory_matches())
    logger.info("- Average time taken to find similar logs: %s", summary.avg_similar_logs_time_taken())
    logger.info("- Average time taken to generate templates: %s", summary.avg_template_generation_time_taken())
    logger.info("- Average total time taken to parse each log: %s", summary.avg_total_time_taken())
