import logging
import shutil
from pathlib import Path

import pandas as pd
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

from rov_parser import config
from rov_parser.parser import Parser
from rov_parser.vector_store import VectorStore

if not Path.exists(Path(config.LOGS_OUT_DIR)):
    Path.mkdir(Path(config.LOGS_OUT_DIR))

log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logging.getLogger().addHandler(console_handler)


# Load the embeddings model
local_embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDINGS_MODEL, model_kwargs={"trust_remote_code": True})

# Reset the chroma database if the flag is set
if config.RESET_CHROMA_DB and Path.exists(Path(config.CHROMA_PERSIST_DIR)):
    shutil.rmtree(config.CHROMA_PERSIST_DIR)

# Create the vector store
vector_store = VectorStore(config.CHROMA_PERSIST_DIR, local_embeddings)

# Create the parser model
parser_pipeline = HuggingFacePipeline.from_model_id(
    model_id=config.PARSER_MODEL,
    task="text-generation",
    device_map="auto",
    pipeline_kwargs={"temperature": config.PARSER_TEMPERATURE, "max_length": config.PARSER_NUM_CTX, "truncation": True},
)
parser_model = ChatHuggingFace(llm=parser_pipeline)

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
