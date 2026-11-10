from src.document_qa.vector_store import VectorStoreManager
from src.document_qa.prompt_builder import PromptBuilder
from langchain_ollama import ChatOllama

class RAGPipeline:
    def __init__(
            self,
            chroma_path: str,
            embedding_model: str,
            llm_model: str,
            top_k: int = 3
    ):
        self.chroma_path = chroma_path
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.top_k = top_k

        self.vector_store = VectorStoreManager(
            persist_directory = self.chroma_path,
            embedding_model = self.embedding_model
        ).load_vector_store()

        self.llm = ChatOllama(model = self.llm_model)
        print(f"[INFO] LLM model: {(self.llm_model)} initialized")

        self.prompt = PromptBuilder.build()

    def retrieve_chunks(self, question: str):
        search_results = self.vector_store.similarity_search(question, k= self.top_k)
        print(f"[DEBUG] Retrieved chunkssize {(len(search_results))} for question {(question)}")

        return search_results

    def build_context(self, documents):
        context = "\n\n ---\n\n".join([doc.page_content for doc in documents])
        print(f"[DEBUG] Context generated: {(context)}")
        return context

    def query(self, question:str):
        print(f"RAG Query Initiated: {(question)}")
        documents = self.retrieve_chunks(question)
       # print(f"Chunks filtered:  \n\n {(documents)}")
        context = self.build_context(documents)
        # formatted_prompt = self.prompt.format(context = context, question = question)
        
        prompt_value = self.prompt.invoke({
            "context": context,
            "question":question
        })
        print("prompt_value", prompt_value)
        response = self.llm.invoke(prompt_value)
        
        print(f"\n\n\n---------- \n\n\nGenerated response: {(response.content)}")

        return{
            "question": question,
             "answer": response.content,
             "documents": documents
         }