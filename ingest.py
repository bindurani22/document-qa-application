from config import DIR_PATH, CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_PATH, EMBEDDING_MODEL
from src.document_qa.document_loader import  DocumentLoader
from src.document_qa.text_chunker import TextChunker
from src.document_qa.vector_store import VectorStoreManager


def run_ingestion(file_path=None):

    file_path =file_path or DIR_PATH

    loader = DocumentLoader(file_path)
    documents = loader.load_documents()
    print(f"[DEBUG] - Ingest.py - length of documents ::  {len(documents)}")

    chunker = TextChunker(chunk_size= CHUNK_SIZE, chunk_overlap= CHUNK_OVERLAP)
    chunks = chunker.chunk_documents(documents)
    print(f"[DEBUG] - Ingest.py - length of chunks :: {len(chunks)}")

    vector_store = VectorStoreManager(
                    persist_directory = CHROMA_PATH,
                    embedding_model = EMBEDDING_MODEL)
    chroma = vector_store.create_vector_store("RAG", chunks)
    print(f"[DEBUG] - Ingest.py - chroma data :: {chroma}")
    print(f"Chroma collection_name: {(chroma._collection_name)}")
    print(f"Chroma embedding_function: {(chroma._embedding_function)}")
    #print(f"Chroma object embedding: {(chroma._)}")
    print(f"Ingested {len(chunks)} chunks into ChromaDB")
    return len(chunks)

def main():
    run_ingestion()

if __name__ == "__main__":
    main()

