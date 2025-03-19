from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable


class Backend(ABC):
    """Abstract base class for backend implementations.

    This class defines the interface for backend services that provide
    embeddings and parser models.
    """

    @abstractmethod
    def get_embeddings(self, model: str) -> Embeddings:
        """Retrieve embeddings from the specified model.

        Args:
            model (str): The name or identifier of the model from which to get embeddings.

        Returns:
            Embeddings: The embeddings retrieved from the specified model.

        """

    @abstractmethod
    def get_parser_model(self, model: str, temperature: float) -> Runnable:
        """Retrieve a parser model based on the specified parameters.

        Args:
            model (str): The name or identifier of the model to retrieve.
            temperature (float): The temperature parameter for the model, which controls the randomness of the output.

        Returns:
            Runnable: An instance of a Runnable object that represents the parser model.

        """


class HuggingFaceBackend(Backend):
    """A backend implementation that uses Hugging Face models for generating embeddings and parsing text."""

    def get_embeddings(self, model: str) -> Embeddings:
        try:
            from langchain_huggingface.embeddings import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(model_name=model, model_kwargs={"trust_remote_code": True})
        except ModuleNotFoundError as e:
            msg = "Please install langchain-huggingface to use HuggingFaceBackend"
            raise ImportError(msg) from e

    def get_parser_model(self, model: str, temperature: float) -> Runnable:
        try:
            from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

            parser_pipeline = HuggingFacePipeline.from_model_id(
                model_id=model,
                task="text-generation",
                device_map="auto",
                pipeline_kwargs={
                    "temperature": temperature,
                },
            )
            return ChatHuggingFace(llm=parser_pipeline)
        except ModuleNotFoundError as e:
            msg = "Please install langchain-huggingface to use HuggingFaceBackend"
            raise ImportError(msg) from e


class OllamaBackend(Backend):
    """A backend implementation that uses Ollama models for generating embeddings and parsing text."""

    def get_embeddings(self, model: str) -> Embeddings:
        try:
            from langchain_ollama.embeddings import OllamaEmbeddings  # type: ignore[import]

            return OllamaEmbeddings(model=model)
        except ModuleNotFoundError as e:
            msg = "Please install langchain-ollama to use OllamaBackend"
            raise ImportError(msg) from e

    def get_parser_model(self, model: str, temperature: float) -> Runnable:
        try:
            from langchain_ollama.chat_models import ChatOllama  # type: ignore[import]

            return ChatOllama(model=model, temperature=temperature, num_ctx=1024*12)
        except ModuleNotFoundError as e:
            msg = "Please install langchain-ollama to use OllamaBackend"
            raise ImportError(msg) from e
