import os

from dotenv import load_dotenv

load_dotenv()

# Whether to use the Ollama backend for parsing logs.
# The default is to use the HuggingFace backend instead.
USE_OLLAMA_BACKEND = bool(int(os.getenv("USE_OLLAMA_BACKEND", "0")))

# The huggingface hub api token to use for downloading models,
# generated from https://huggingface.co/docs/hub/security-tokens.
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", None)

if not HUGGINGFACEHUB_API_TOKEN and not USE_OLLAMA_BACKEND:
    msg = "HUGGINGFACEHUB_API_TOKEN is not set"
    raise ValueError(msg)

# The input path to the logs to parse.
TEST_LOG_PATH = os.getenv("TEST_LOG_PATH", "data/test.csv")

# The output path to save the parsed logs with templates.
TEST_OUT_PATH = os.getenv("TEST_OUT_PATH", "data/test_out.csv")

# The path to the dir where the execution logs will be saved.
LOGS_OUT_DIR = os.getenv("LOGS_OUT_PATH", "logs")

# The path to the dir where the chroma database should be stored.
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")

# Whether to reset the chroma database on startup.
RESET_CHROMA_DB = bool(int(os.getenv("RESET_CHROMA_DB", "0")))

# The model used to embed logs.
# Must be a model from the HuggingFace model hub if using the HuggingFace backend,
# or a model from the Ollama model hub if using the Ollama backend.
EMBEDDINGS_MODEL = os.getenv(
    "EMBEDDINGS_MODEL",
    "nomic-embed-text" if USE_OLLAMA_BACKEND else "nomic-ai/nomic-embed-text-v1.5",
)

# The model used to parse logs.
# Must be a model from the HuggingFace model hub if using the HuggingFace backend,
# or a model from the Ollama model hub if using the Ollama backend.
PARSER_MODEL = os.getenv(
    "PARSER_MODEL",
    "deepseek-r1:7b" if USE_OLLAMA_BACKEND else "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
)

# The temperature of the LLM used to parse logs.
# Must be between 0 and 1.
# The recommended value for DeepSeek R1 is 0.6.
PARSER_TEMPERATURE = float(os.getenv("PARSER_TEMPERATURE", "0.6"))

if PARSER_TEMPERATURE < 0 or PARSER_TEMPERATURE > 1:
    msg = "PARSER_TEMPERATURE must be between 0 and 1"
    raise ValueError(msg)

# The number of context tokens to use when parsing logs.
PARSER_NUM_CTX = int(os.getenv("PARSER_NUM_CTX", "4096"))

# The minimum amount of quality (defined as number of very similar logs)
# that a log must have to be considered a memory match.
# Must be greater than 0.
MEMORY_MATCH_MIN_QUALITY = int(os.getenv("MEMORY_MATCH_MIN_QUALITY", "3"))

if MEMORY_MATCH_MIN_QUALITY <= 0:
    msg = "MEMORY_MATCH_MIN_QUALITY must be greater than 0"
    raise ValueError(msg)

# The number of self-reflection steps to take.
# Must be greater than 0.
SELF_REFLECTION_STEPS = int(os.getenv("SELF_REFLECTION_STEPS", "2"))

if SELF_REFLECTION_STEPS <= 0:
    msg = "SELF_REFLECTION_STEPS must be greater than 0"
    raise ValueError(msg)
