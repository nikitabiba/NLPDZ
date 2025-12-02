from typing import List, Dict, Any, Generator
from langchain.schema import Document
from langchain_gigachat.chat_models import GigaChat
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import time


class LLM:
    def __init__(self, gigachat_token: str, model: str = "GigaChat-2-Max", max_history: int = 6):
        self.gigachat_token = gigachat_token
        self.model = model
        self.max_history = max_history
        self.history: List[Dict[str, str]] = []

        self.system_prompt = """
# Роль
Ты — AI-ассистент для работы с кодовой базой проекта. Твоя задача — помогать разработчикам понимать код, находить информацию и решать задачи на основе предоставленного контекста из репозитория.

# КРИТИЧЕСКИ ВАЖНО: Проверка релевантности
**ПЕРВЫМ ДЕЛОМ** определи, относится ли вопрос к программированию, кодовой базе или разработке ПО.
**ВТОРЫМ ДЕЛОМ** не придумывай контекст. Бери код только из предоставленного контекста и ни откуда больше.

## Вопросы НЕ по теме:
- Кулинарные рецепты
- Медицинские советы
- Туристические маршруты
- Общие знания не связанные с кодом
- Любые темы, не относящиеся к разработке

**Если вопрос НЕ по теме → ответь ТОЛЬКО:**
```
Я могу помочь только с вопросами, связанными с кодовой базой проекта.
```
**И больше НИЧЕГО не добавляй. Не объясняй, не извиняйся, не предлагай альтернатив.**

## Вопросы ПО ТЕМЕ:
- Как работает код/функция/модуль?
- Где находится реализация X?
- Как добавить/изменить функциональность?
- Объяснение архитектуры
- Отладка и поиск багов
- Вопросы о зависимостях, конфигурации, API
- Общие вопросы о программировании в контексте проекта

**Только для вопросов ПО ТЕМЕ → продолжай по инструкциям ниже.**

---

# Контекст
Тебе будет предоставлен релевантный контекст из кодовой базы, который может включать:
- Фрагменты исходного кода
- Документацию
- Комментарии и docstrings
- Историю коммитов
- README и другие markdown файлы

# Инструкции

## Анализ кода
1. **Основывайся только на предоставленном контексте** — не придумывай детали реализации
2. **Указывай источники** — ссылайся на конкретные файлы и строки кода
3. **Будь точным** — если информации недостаточно, честно скажи об этом
4. **Объясняй архитектуру** — показывай связи между компонентами

## Ответы на вопросы
- Для вопросов "как работает X?" — объясни логику с примерами из кода и указаним путей к коду
- Для вопросов "где находится X?" — укажи точные пути к файлам
- Для вопросов "почему X сделано так?" — ищи комментарии и документацию
- Указывай полные пути к файлам при ответе на вопрос по коду
- Если нужной информации нет в контексте, предложи, где её можно найти

## Помощь с задачами
- При запросах на изменения — анализируй существующие паттерны в коде
- Предлагай решения, совместимые с текущей архитектурой
- Указывай на потенциальные побочные эффекты
- Ссылайся на похожие реализации в кодовой базе

## Формат ответа
- Отвечай конкретно и чётко по конкретному вопросу
- Выделяй важные моменты

# Ограничения
- НЕ предлагай код, противоречащий стилю проекта
- НЕ делай предположений о коде вне предоставленного контекста
- НЕ ВРИ!
- Если вопрос требует информации из нескольких частей системы, которые не представлены в контексте, укажи это явно



------------ ИСТОРИЯ ДИАЛОГА ------------
{history}

------------ КОНТЕКСТ ------------
{context}

------------ ВОПРОС ------------
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

    def _trim_history(self):
        """Ограничивает историю диалога."""
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def _format_document(self, doc: Document) -> Document:
        """Форматирование документа в человекочитаемый вид."""
        meta = doc.metadata

        def fmt(key, default="—"):
            val = meta.get(key, default)
            if isinstance(val, list):
                str_items = []
                for item in val:
                    if isinstance(item, dict):
                        if 'name' in item:
                            str_items.append(str(item['name']))
                        else:
                            str_items.append(str(item))
                    else:
                        str_items.append(str(item))
                return ", ".join(str_items) if str_items else "—"
            elif isinstance(val, dict):
                if 'name' in val:
                    return str(val['name'])
                else:
                    return str(val)
            return str(val) if val else "—"

        formatted = f"""
[FILE: {fmt("source_file")} — chunk {fmt("chunk_index")}]

Language: {fmt("language")}
Classes: {fmt("classes")}
Functions: {fmt("functions")}
Imports: {fmt("imports")}
Clean Imports: {fmt("clean_imports")}
Calls: {fmt("calls")}
Docstrings: {fmt("docstrings")}

--- CODE ---
{doc.page_content}
""".strip()

        return Document(page_content=formatted, metadata=doc.metadata)

    def _build_context(self, docs: List[Document]) -> List[Document]:
        """Форматирует документы для подачи в модель."""
        return [self._format_document(doc) for doc in docs]

    def _build_history_text(self) -> str:
        """Собирает историю сообщений в текст для промпта."""
        return "\n".join(
            f"{'User' if msg['role']=='user' else 'Assistant'}: {msg['content']}"
            for msg in self.history
        )

    def ask(self, docs: List[Document], question: str) -> str:
        """Обычный синхронный запрос."""
        formatted_docs = self._build_context(docs)
        history_text = self._build_history_text()

        response = self.document_chain.invoke({
            "input": question,
            "history": history_text,
            "context": formatted_docs
        })

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": response})
        self._trim_history()

        return response

    def generate_response_streaming(self, docs: List[Document], question: str) -> Generator[str, None, None]:
        """
        "Stream" версия — отдаёт текст частями.
        GigaChat не даёт stream, поэтому делаем по фрагментам.
        """
        answer = self.ask(docs, question)

        chunk_size = 120
        for i in range(0, len(answer), chunk_size):
            yield answer[i:i + chunk_size]
            time.sleep(0.01)

    def generate_response(self, docs: List[Document], question: str) -> str:
        answer = ""
        for part in self.generate_response_streaming(docs, question):
            answer += part
        return answer

    def clear_history(self):
        self.history = []