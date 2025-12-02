from src.Retriever import Retriever
from src.LLM import LLM
import logging
from langchain_community.embeddings import HuggingFaceEmbeddings


class RAG:
    def __init__(self, model_name, json_data_path, index_file_name, vector_store_path, gigachat_token=None, llm_model="GigaChat-2-Max", retrieval_k=10):

        self.retrieval_k = retrieval_k

        print("Инициализация ретривера...")
        self.retriever = Retriever(
            emdedding_model=HuggingFaceEmbeddings(model_name=model_name),
            json_data_path=json_data_path,
            index_file_name=index_file_name,
            vector_store_path=vector_store_path
        )

        print("Инициализация LLM...")
        self.llm = LLM(gigachat_token=gigachat_token, model=llm_model)

        print("Загрузка данных...")
        self.retriever.load_data()
        print("RAG-ассистент готов к работе!")

    def ask(self, question):
        """
        Задать вопрос ассистенту (синхронная версия).

        Returns:
            {
                "answer": str,
                "sources": list[dict],
                "retrieved_docs_count": int
            }
        """
        try:
            print(f"Поиск релевантных документов для запроса: '{question}'")
            retrieved_docs = self.retriever.retrieve(question, self.retrieval_k)

            print(f"Найдено {len(retrieved_docs)} релевантных документов")

            sources = []
            for doc in retrieved_docs:
                sources.append({
                    "раздел": doc.metadata.get("раздел", "Неизвестно"),
                    "глава": doc.metadata.get("глава", "Неизвестно"),
                    "статья": doc.metadata.get("статья", "Неизвестно"),
                    "пункт": doc.metadata.get("пункт", "Неизвестно"),
                    "текст": doc.page_content
                })

            print("Генерация ответа...")

            answer = self.llm.ask(retrieved_docs, question)

            return {
                "answer": answer,
                "sources": sources,
                "retrieved_docs_count": len(retrieved_docs)
            }

        except Exception as e:
            logging.error(f"Ошибка при обработке запроса: {e}")
            return {
                "answer": f"Произошла ошибка при обработке вашего запросa: {str(e)}",
                "sources": [],
                "retrieved_docs_count": 0
            }

