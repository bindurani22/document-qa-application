from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap  = chunk_overlap


    def chunk_documents(self, documents):
        splitter = RecursiveCharacterTextSplitter(
                chunk_size = self.chunk_size,
                chunk_overlap = self.chunk_overlap
        )

        chunks = splitter.split_documents(documents)

        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks")

        # for i, chunk in enumerate(chunks):
        #     print(f"\n--- CHUNK{i} ---")
        #     print(f"Chunk Pagecontent: {(chunk.page_content)}")                  
        #     print(f"Chunk Metadata: {(chunk.metadata)}")

        return chunks