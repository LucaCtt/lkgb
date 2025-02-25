from abc import ABC, abstractmethod

from chromadb import Embeddings
from langchain_core.runnables import Runnable
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


class Backend(ABC):
    @abstractmethod
    def get_embeddings(self, model: str) -> Embeddings:
        pass

    @abstractmethod
    def get_parser_model(self, model: str, temperature: float, context_length: int) -> Runnable:
        pass


class HuggingFaceBackend(Backend):
    def get_embeddings(self, model: str) -> Embeddings:
        return HuggingFaceEmbeddings(model_name=model, model_kwargs={"trust_remote_code": True})

    def get_parser_model(self, model: str, temperature: float, context_length: int) -> Runnable:
        parser_pipeline = HuggingFacePipeline.from_model_id(
            model_id=model,
            task="text-generation",
            device_map="auto",
            pipeline_kwargs={
                "temperature": temperature,
                "max_length": context_length,
                "truncation": True,
            },
        )
        return ChatHuggingFace(llm=parser_pipeline)


class OllamaBackend(Backend):
    def __init__(self) -> "OllamaBackend":
        super().__init__()

    def get_embeddings(self, model: str) -> Embeddings:
        try:
            from langchain_ollama.embeddings import OllamaEmbeddings  # type: ignore[import]

            return OllamaEmbeddings(model=model)
        except ModuleNotFoundError as e:
            msg = "Please install langchain-ollama to use OllamaBackend"
            raise ImportError(msg) from e

    def get_parser_model(self, model: str, temperature: float, context_length: int) -> Runnable:
        try:
            from langchain_ollama.chat_models import ChatOllama  # type: ignore[import]

            return ChatOllama(model=model, temperature=temperature, num_ctx=context_length)
        except ModuleNotFoundError as e:
            msg = "Please install langchain-ollama to use OllamaBackend"
            raise ImportError(msg) from e
