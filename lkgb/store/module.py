from abc import ABC, abstractmethod

from lkgb.config import Config


class StoreModule(ABC):
    """Abstract class for store modules."""

    def __init__(self, config: Config) -> None:
        self._config = config

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the store module by creating module-specific nodes."""

    @abstractmethod
    def clear(self) -> None:
        """Clear all data in the store module.

        This method should delete only the nodes specific to this module.
        """
