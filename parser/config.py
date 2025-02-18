import os

from dotenv import load_dotenv

load_dotenv()

TEST_LOG_PATH = os.getenv("TEST_LOG_PATH", "test_data/with_template.csv")

# The ollama model used to embed logs
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text")

# The ollama model used to parse logs
PARSER_MODEL = os.getenv("PARSER_MODEL", "qwen2.5-coder:7b")

# The temperature of the LLM used to parse logs
PARSER_TEMPERATURE = float(os.getenv("PARSER_TEMPERATURE", "0.5"))

# The number of self-reflection steps to take.
# If less or equal to 0, no self-reflection is done
SELF_REFLECTION_STEPS = int(os.getenv("SELF_REFLECTION_STEPS", "3"))
