from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

class VectorStoreManager:
    def __init__(self, persist_directory: str, embedding_model: str):
        self.persist_directory =  persist_directory
        self.embedding_model = embedding_model
        self.embeddings = OllamaEmbeddings(model = self.embedding_model)

    def create_vector_store(self, collection_name, documents):
        vectorstore = Chroma.from_documents(
                collection_name = collection_name,
                documents = documents,
                embedding = self.embeddings,
                persist_directory = self.persist_directory
        )
        return vectorstore

    def load_vector_store(self):
        return Chroma(
            persist_directory = self.persist_directory,
            embedding_function = self.embeddings
        )