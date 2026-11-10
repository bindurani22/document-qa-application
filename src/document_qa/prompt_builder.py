from langchain_core.prompts import ChatPromptTemplate

class PromptBuilder:
    @staticmethod
    def build():
        chatprompt = ChatPromptTemplate.from_template("""
        Answer the question based only on the context below.

        If the answer is not in the context, say:
        "I oculd not find the answer in the provided document."

        Context:
        {context}

        Question:
        {question}
        """)

        return chatprompt