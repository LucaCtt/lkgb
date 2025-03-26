from langchain_core.embeddings import Embeddings

from lkgb.config import Config
from lkgb.store.dataset import Dataset
from lkgb.store.driver import Driver
from lkgb.store.ontology import Ontology


class Store:
    def __init__(self, config: Config, embeddings: Embeddings) -> "Store":
        self.__config = config
        self.__embeddings = embeddings

        self.driver = Driver(config.neo4j_url, config.neo4j_username, config.neo4j_password)
        self.ontology = Ontology(self.driver)
        self.dataset = Dataset(self.driver, self.__embeddings)

    def initialize(self) -> None:
        self.driver.initialize(self.__config)
        self.ontology.initialize(self.__config)
        self.dataset.initialize(self.__config)

    def clear(self) -> None:
        self.driver.clear()
        self.ontology.clear()
        self.dataset.clear()
