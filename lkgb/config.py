import os

from dotenv import load_dotenv

load_dotenv()

# Whether to use the Ollama backend for parsing logs.
# The default is to use the HuggingFace backend instead.
USE_OLLAMA_BACKEND = bool(int(os.getenv("USE_OLLAMA_BACKEND", "1")))

# The huggingface hub api token to use for downloading models,
# generated from https://huggingface.co/docs/hub/security-tokens.
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", None)

# The input path to the logs to parse.
TEST_LOG_PATH = os.getenv("TEST_LOG_PATH", "data/test.csv")

# Neo4j config
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# The model used to embed logs.
# Must be a model from the HuggingFace model hub if using the HuggingFace backend,
# or a model from the Ollama model hub if using the Ollama backend.
EMBEDDINGS_MODEL = os.getenv(
    "EMBEDDINGS_MODEL",
    "snowflake-arctic-embed:110m" if USE_OLLAMA_BACKEND else "Snowflake/snowflake-arctic-embed-m",
)

# The model used to parse logs.
# Must be a model from the HuggingFace model hub if using the HuggingFace backend,
# or a model from the Ollama model hub if using the Ollama backend.
PARSER_MODEL = os.getenv(
    "PARSER_MODEL",
    "qwen2.5-coder:7b" if USE_OLLAMA_BACKEND else "Qwen/Qwen2.5-Coder-7B-Instruct",
)

# The temperature of the LLM used to parse logs.
# Must be between 0 and 1.
# The recommended value for DeepSeek R1 is 0.6.
PARSER_TEMPERATURE = float(os.getenv("PARSER_TEMPERATURE", "0.6"))

if PARSER_TEMPERATURE < 0 or PARSER_TEMPERATURE > 1:
    msg = "PARSER_TEMPERATURE must be between 0 and 1"
    raise ValueError(msg)

# The number of self-reflection steps to take.
# Must be greater than 0.
SELF_REFLECTION_STEPS = int(os.getenv("SELF_REFLECTION_STEPS", "2"))

if SELF_REFLECTION_STEPS <= 0:
    msg = "SELF_REFLECTION_STEPS must be greater than 0"
    raise ValueError(msg)
