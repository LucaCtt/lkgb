"""Configuration module for setting up variables used in the log graph building.

The variables are set using environment variables, with default values provided.
Environment variables can be set in the shell before running the script, or in a `.env` file
in the root directory of the project.
"""

import hashlib
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _compute_file_hash(file_path: str) -> str:
    """Compute the SHA256 hash of a file."""
    with Path(file_path).open("rb", buffering=0) as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class Config:
    """Configuration class for setting up variables used in the log graph building.

    Having a class for this is useful for easily exporting the configuration as a dictionary.
    """

    # The date and time of the experiment.
    experiment_date_time = datetime.now(UTC)

    # Used to distinguish between the data in different experiments.
    experiment_id = os.getenv("EXPERIMENT_ID", str(uuid.uuid4()))

    # The path to the ontology file.
    ontology_path = os.getenv("ONTOLOGY_PATH", "resources/ontologies/logs_dictionary.ttl")

    # The path to the examples log graphs file.
    examples_path = os.getenv("EXAMPLES_PATH", "resources/data/train.ttl")

    # The input path to the logs to parse.
    tests_path = os.getenv("TEST_LOG_PATH", "resources/data/test.ttl")

    # The prompt used to build the graph
    prompt_build_graph = os.getenv("PROMPT_BUILD_GRAPH", Path("resources/prompts/build_graph.system.md").read_text())

    # Neo4j config
    neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

    # Whether to use the Ollama or HuggingFace backends for parsing logs.
    # The default is to use Ollama.
    use_ollama_backend = bool(int(os.getenv("USE_OLLAMA_BACKEND", "1")))

    # The HuggingFace hub api token to use for downloading models,
    # generated from https://huggingface.co/docs/hub/security-tokens.
    # Only useful with the HuggingFace backend and when using private models.
    # For public models, this can be left unset.
    # This variable is not used anywhere in the project,
    # it's just to remind that it can be set in the environment.
    huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", None)

    # The model used to embed logs.
    # Must be a model from the HuggingFace model hub if using the HuggingFace backend,
    # or a model from the Ollama model hub if using the Ollama backend.
    embeddings_model = os.getenv(
        "EMBEDDINGS_MODEL",
        "snowflake-arctic-embed:110m" if use_ollama_backend else "Snowflake/snowflake-arctic-embed-m",
    )

    # The model used to parse logs.
    # Must be a model from the HuggingFace model hub if using the HuggingFace backend,
    # or a model from the Ollama model hub if using the Ollama backend.
    parser_model = os.getenv(
        "PARSER_MODEL",
        "qwen2.5-coder:14b" if use_ollama_backend else "Qwen/Qwen2.5-Coder-14B-Instruct",
    )

    # The temperature of the LLM used to parse logs.
    # Must be between (strictly) 0 and 1.
    parser_temperature = float(os.getenv("PARSER_TEMPERATURE", "0.5"))

    # The number of self-reflection steps to take.
    # Must be greater or equal than 0.
    self_reflection_steps = int(os.getenv("SELF_REFLECTION_STEPS", "3"))

    def __init__(self):
        if self.parser_temperature < 0 or self.parser_temperature > 1:
            msg = "parser_temperature must be between 0 and 1"
            raise ValueError(msg)

        if self.self_reflection_steps < 0:
            msg = "self_reflection_steps must be greater than 0"
            raise ValueError(msg)

    def ontology_hash(self) -> str:
        return _compute_file_hash(self.ontology_path)

    def examples_hash(self) -> str:
        return _compute_file_hash(self.examples_path)

    def dump(self) -> dict[str, Any]:
        """Dump the configuration as a dictionary.

        This method also includes the ontology and examples hash.
        Authentication params are not included in the dump for security reasons.
        """
        excluded_prefixes = ["_", "neo4j", "huggingface_api_token"]

        dump = {
            key: value
            for key, value in self.__class__.__dict__.items()
            if not any(key.startswith(prefix) for prefix in excluded_prefixes) and not callable(value)
        }
        dump["ontology_hash"] = self.ontology_hash()
        dump["examples_hash"] = self.examples_hash()

        return dump
