from langchain_core.embeddings import Embeddings

from lkgb.config import Config
from lkgb.store.dataset import Dataset
from lkgb.store.driver import Driver
from lkgb.store.module import StoreModule
from lkgb.store.ontology import Ontology


class Store(StoreModule):
    def __init__(self, config: Config, embeddings: Embeddings) -> None:
        super().__init__(config)
        self.__embeddings = embeddings

        self.driver = Driver(self._config)
        self.ontology = Ontology(
            self._config,
            self.driver,
        )
        self.dataset = Dataset(self._config, self.driver, self.__embeddings)

    def initialize(self) -> None:
        self.driver.initialize()
        self.ontology.initialize()
        self.dataset.initialize()

    def clear(self) -> None:
        self.driver.clear()
        self.ontology.clear()
        self.dataset.clear()
