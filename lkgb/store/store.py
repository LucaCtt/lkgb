from langchain_core.embeddings import Embeddings

from lkgb.config import Config
from lkgb.store.dataset import Dataset
from lkgb.store.driver import Driver
from lkgb.store.ontology import Ontology


class Store:
    def __init__(self, config: Config, embeddings: Embeddings) -> "Store":
        self.__config = config
        self.__embeddings = embeddings

        self.driver = Driver(self.__config)
        self.ontology = Ontology(self.driver, self.__config)
        self.dataset = Dataset(self.driver, self.__embeddings, self.__config)

    def initialize(self) -> None:
        self.driver.initialize()
        self.ontology.initialize()
        self.dataset.initialize()

    def clear(self) -> None:
        self.driver.clear()
        self.ontology.clear()
        self.dataset.clear()
