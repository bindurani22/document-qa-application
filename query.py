from config import CHROMA_PATH, EMBEDDING_MODEL, LLM_MODEL, TOP_K
from src.document_qa.rag_pipeline import RAGPipeline

def run_query(question: str):
    pipeline = RAGPipeline(
        chroma_path=CHROMA_PATH,
        embedding_model=EMBEDDING_MODEL,
        llm_model=LLM_MODEL,
        top_k=TOP_K
    )

    question = "What is mutual fund?" #input("Enter your question:") 
    print(question)
    result = pipeline.query(question)

    print(result)
    return result

def main():
    question = "Tell about mutual fund?"
    print("Ingest main: question:",question)
    result = run_query(question)
    print("Ingest main: result:",result)

if __name__ == "__main__":
    main()