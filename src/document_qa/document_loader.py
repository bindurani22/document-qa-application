from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader

class DocumentLoader:
    def  __init__(self, dir_path: str):
        self.dir_path = dir_path

    def load_documents(self) -> List[Any]:

        documents = []
        path = Path(self.dir_path).resolve()
        print(f"[DEBUG] Directory path: {path}")

        pdf_files = list(path.glob('**/*.pdf'))
        print(f"[DEBUG] No. of pdf file found: {len(pdf_files)}")
        print(f"[DEBUG] PDF Files found: {[str(i) for i in pdf_files]}")

        for pdf_file in pdf_files:
            try: 
                loader = PyPDFLoader(pdf_file)
                pdf_content = loader.load()

                print(f"[DEBUG] loaded {len(pdf_content)} PDF files")

                documents.extend(pdf_content)

            except Exception as e:
                print(f"[ERROR] Failed loading {(pdf_file)}: e")

        return documents