import os

from dotenv import load_dotenv

load_dotenv()

TEST_LOG_PATH = os.getenv("TEST_LOG_PATH", "data/hadoop_2k.csv")

# The path to the dir where the chroma database should be stored
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")

# The ollama model used to embed logs
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text")

# The ollama model used to parse logs
PARSER_MODEL = os.getenv("PARSER_MODEL", "qwen2.5-coder:7b")

# The temperature of the LLM used to parse logs.
# Must be between 0 and 1.
PARSER_TEMPERATURE = float(os.getenv("PARSER_TEMPERATURE", "0.5"))

# The number of self-reflection steps to take.
# Must be greater than 0.
SELF_REFLECTION_STEPS = int(os.getenv("SELF_REFLECTION_STEPS", "3"))
