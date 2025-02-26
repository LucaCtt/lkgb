import logging
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from rov_parser import config
from rov_parser.backend import HuggingFaceBackend, OllamaBackend
from rov_parser.parser import Parser
from rov_parser.vector_store import VectorStore

if not Path.exists(Path(config.LOGS_OUT_DIR)):
    Path.mkdir(Path(config.LOGS_OUT_DIR))

log_formatter = logging.Formatter("%(asctime)s [%(levelname)-4.4s]  %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger = logging.getLogger()
logger.addHandler(console_handler)

if config.USE_OLLAMA_BACKEND:
    logger.info("Using Ollama backend")
    backend = OllamaBackend()
else:
    logger.info("Using HuggingFace backend")
    backend = HuggingFaceBackend()

# Load the embeddings model
local_embeddings = backend.get_embeddings(model=config.EMBEDDINGS_MODEL)

# Reset the chroma database if the flag is set
if config.RESET_CHROMA_DB and Path.exists(Path(config.CHROMA_PERSIST_DIR)):
    shutil.rmtree(config.CHROMA_PERSIST_DIR)

# Create the vector store
vector_store = VectorStore(config.CHROMA_PERSIST_DIR, local_embeddings)

# Create the parser model
parser_model = backend.get_parser_model(
    model=config.PARSER_MODEL,
    temperature=config.PARSER_TEMPERATURE,
    context_length=config.PARSER_NUM_CTX,
)

parser = Parser(parser_model, vector_store, config.MEMORY_MATCH_MIN_QUALITY, config.SELF_REFLECTION_STEPS)

if __name__ == "__main__":
    logs_df = pd.read_csv(config.TEST_LOG_PATH)
    logs_df = logs_df.fillna("")

    for log in tqdm(logs_df["text"], desc="Processing logs"):
        parser.compute_template(log)

    with Path.open(config.TEST_OUT_PATH, "w") as out_file:
        out_file.write("text,template\n")
        for log in tqdm(logs_df["text"], desc="Writing output"):
            template = vector_store.get_template(log)
            out_file.write(f"{log},{template}\n")
