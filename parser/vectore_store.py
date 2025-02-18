from chromadb import Embeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorStore:
    """
    VectorStore is thin wrapper around Chroma for storing and retrieving text embeddings.
    """

    def __init__(self, persist_dir: str, embeddings_model: Embeddings) -> "VectorStore":
        """
        Initializes the VectorStore with a persistent directory and an embeddings model.

        Args:
            persist_dir (str): The directory where the vector store will persist its data.
            embeddings_model (Embeddings): The model used to generate embeddings for the data.

        Attributes:
            store (Chroma): The Chroma vector store initialized with the given directory and embeddings model.
            splitter (RecursiveCharacterTextSplitter): A text splitter that splits text into chunks of size 1000 with an overlap of 100 characters.

        """
        self.store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings_model,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    def load_csv(self, csv_path: str) -> None:
        """
        Loads data from a CSV file, splits the data into chunks, and adds the chunks to the store.

        Args:
            csv_path (str): The path to the CSV file to be loaded.

        The CSV file is expected to have the following columns:
            - line_number
            - tactic
            - techniques
            - file_name
            - template

        The function performs the following steps:
            1. Loads the CSV file using CSVLoader with specified metadata columns.
            2. Splits the loaded data into chunks using the splitter.
            3. Adds the chunks to the store.

        """
        loader = CSVLoader(
            file_path=csv_path,
            metadata_columns=["line_number", "tactic", "techniques", "file_name", "template"],
        )
        data = loader.load()

        # Split the logs into chunks
        all_splits = self.splitter.split_documents(data)

        self.store.add_documents(all_splits)

    def find_very_similar_logs_with_template(self, log: str) -> list[Document]:
        """
        Finds logs that are very similar to the given log using a predefined template.

        This method searches for logs in the store that have a high similarity score
        with the provided log. It uses a similarity threshold and filters out logs
        that do not have a template.

        Args:
            log (str): The log string to find similar logs for.

        Returns:
            list[Document]: A list of Document objects that are very similar to the given log.

        """
        similar = self.store.similarity_search_with_relevance_scores(
            self.__compose_similarity_question(log),
            score_threshold=0.7,
            k=10,
            filter={"template": {"$ne": ""}},
        )
        return [doc for doc, _ in similar]

    def find_similar_logs(self, log: str) -> list[Document]:
        """
        Find logs similar to the given log.

        This method searches for logs that are similar to the provided log string
        using a similarity search algorithm. It returns a list of Document objects
        that are considered similar based on a relevance score threshold.

        Args:
            log (str): The log string to find similar logs for.

        Returns:
            list[Document]: A list of Document objects that are similar to the given log.

        """
        similar = self.store.similarity_search_with_relevance_scores(
            self.__compose_similarity_question(log),
            score_threshold=0.4,
            k=5,
        )
        return [doc for doc, _ in similar]

    def add_document(self, document: Document) -> None:
        """
        Adds a new document to the store.

        Args:
            document (Document): The document object to be added to the store.

        """
        all_splits = self.splitter.split_documents([document])
        self.store.add_documents(all_splits)

    def update_document(self, document: Document) -> None:
        """
        Updates an existing document in the store.

        Args:
            document (Document): The document object containing updated information.

        """
        self.store.update_document(document_id=document.id, document=document)

    def __compose_similarity_question(self, log: str) -> str:
        return f'Which logs are most similar to "{log}"?'
