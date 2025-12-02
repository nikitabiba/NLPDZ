from langchain.schema import Document
from langchain_gigachat.chat_models import GigaChat
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain


class LLM:
    def __init__(self, gigachat_token, model = "GigaChat-2-Max"):
        self.gigachat_token = gigachat_token
        self.model = model

        self.system_prompt = """
# Роль
Ты — интеллектуальный помощник по Конституции Российской Федерации. 
Твоя задача — отвечать на вопросы пользователя, используя только предоставленный контекст 
(из статей, глав и разделов Конституции РФ).

# Критически важно
1. **Всегда опирайся только на предоставленный контекст.**
2. **Не придумывай нормы, статьи, формулировки.**
3. **Если в контексте нет нужной информации, скажи прямо:**
В предоставленном контексте нет информации для точного ответа.
4. **Никогда не цитируй Конституцию, если цитата отсутствует в контексте.**
5. **Не используй знания вне переданных документов.**

# Что ты можешь делать
- Отвечать на правовые вопросы строго по тексту Конституции РФ.
- Объяснять смысл статей и норм, если это прямо следует из текста.
- Делать выводы, опираясь на предоставленный фрагмент.
- Суммировать или структурировать данные из контекста.

# Чего делать нельзя
- Не интерпретировать нормы шире, чем позволяет текст.
- Не ссылаться на законы, документы или нормы, отсутствующие в контексте.
- Не давать юридических советов вне рамок Конституции.
- Не рассуждать на темы, не связанные с вопросом пользователя.

# Формат ответа
- Ясный, краткий, основанный на контексте.
- Если возможно — указывай, из каких пунктов/статей контекст взят.
- Не используй художественные домыслы.

-------------------------------------
# КОНТЕКСТ
{context}

-------------------------------------
# ВОПРОС
{input}
"""
        self.llm = GigaChat(
            credentials=gigachat_token,
            verify_ssl_certs=False,
            model=model
        )

        self.prompt = ChatPromptTemplate.from_template(self.system_prompt)

        self.document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=self.prompt
        )

    def _format_document(self, doc):
        """Форматирование документа для модели."""
        meta = doc.metadata

        formatted = f"""
    [СТРУКТУРА]
    Раздел: {meta.get("раздел", "—")}
    Глава: {meta.get("глава", "—")}
    Статья: {meta.get("статья", "—")}
    Пункт: {meta.get("пункт", "—")}

    --- ТЕКСТ ---
    {doc.page_content}
    """.strip()

        return Document(page_content=formatted, metadata=meta)

    def _build_context(self, docs):
        return [self._format_document(doc) for doc in docs]

    def ask(self, docs, question):
        formatted_docs = self._build_context(docs)

        response = self.document_chain.invoke({
            "input": question,
            "context": formatted_docs,
        })

        return response
