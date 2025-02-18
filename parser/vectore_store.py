from chromadb import Embeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_vector_store(logs_path: str, persist_dir: str, embeddings_model: Embeddings) -> Chroma:
    """
    Creates a vector store from log data using the specified embeddings model.

    Args:
        logs_path (str): The file path to the CSV file containing the logs.
        persist_dir (str): The directory where the vector store should be persisted.
        embeddings_model (Embeddings): The embeddings model to use for creating the vector store.

    Returns:
        Chroma: A Chroma vector store created from the log data.

    """
    # Load the logs from the CSV file
    loader = CSVLoader(
        file_path=logs_path,
        metadata_columns=["line_number", "tactic", "techniques", "template"],
    )
    data = loader.load()

    # Split the logs into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)

    # Create the vector store
    return Chroma.from_documents(
        documents=all_splits,
        persist_directory=persist_dir,
        embedding=embeddings_model,
        collection_metadata={"hnsw:space": "cosine"},
    )
