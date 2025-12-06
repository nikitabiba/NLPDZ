import streamlit as st
from src.RAG import RAG
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

st.set_page_config(
    page_title="Конституция РФ - Ассистент",
    layout="centered"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_system" not in st.session_state:
    st.session_state.rag_system = None

with st.sidebar:
    st.header("Настройки")
    
    gigachat_token = st.text_input(
        "GigaChat Token",
        type="password",
        help="Введите токен для доступа к GigaChat API"
    )
    
    model_name = st.selectbox(
        "Модель эмбеддингов",
        ["sberbank-ai/sbert_large_nlu_ru"],
        help="Модель для векторизации текста"
    )
    
    llm_model = st.selectbox(
        "Модель LLM",
        ["GigaChat-2-Max", "GigaChat"],
        help="Модель для генерации ответов"
    )
    
    retrieval_k = st.slider(
        "Количество документов",
        min_value=5,
        max_value=20,
        value=10,
        help="Количество извлекаемых релевантных документов"
    )
    
    json_data_path = st.text_input(
        "Путь к JSON данным",
        value="data/json"
    )
    
    vector_store_path = st.text_input(
        "Путь к векторному хранилищу",
        value="data/vector_store"
    )
    
    index_file_name = st.text_input(
        "Имя индекса",
        value="index"
    )
    
    if st.button("Инициализировать систему"):
        if not gigachat_token:
            st.error("Введите токен GigaChat")
        else:
            with st.spinner("Инициализация RAG-системы..."):
                try:
                    st.session_state.rag_system = RAG(
                        model_name=model_name,
                        json_data_path=json_data_path,
                        index_file_name=index_file_name,
                        vector_store_path=vector_store_path,
                        gigachat_token=gigachat_token,
                        llm_model=llm_model,
                        retrieval_k=retrieval_k
                    )
                    st.success("Система готова к работе")
                except Exception as e:
                    st.error(f"Ошибка инициализации: {str(e)}")
    
    if st.button("Очистить историю"):
        st.session_state.messages = []
        st.rerun()

st.title("Ассистент по Конституции РФ")
st.caption("Задайте вопрос о Конституции Российской Федерации")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Источники"):
                for idx, source in enumerate(message["sources"], 1):
                    st.markdown(f"**{idx}. {source['статья']}, {source['пункт']}**")
                    st.text(f"Раздел: {source['раздел']}")
                    st.text(f"Глава: {source['глава']}")
                    st.text(f"Текст: {source['текст'][:200]}...")
                    st.divider()

if prompt := st.chat_input("Введите ваш вопрос"):
    if st.session_state.rag_system is None:
        st.error("Сначала инициализируйте систему в боковой панели")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Поиск информации..."):
                try:
                    response = st.session_state.rag_system.ask(prompt)
                    
                    answer = response["answer"]
                    sources = response["sources"]
                    doc_count = response["retrieved_docs_count"]
                    
                    st.markdown(answer)
                    
                    with st.expander(f"Источники ({doc_count} документов)"):
                        for idx, source in enumerate(sources, 1):
                            st.markdown(f"**{idx}. {source['статья']}, {source['пункт']}**")
                            st.text(f"Раздел: {source['раздел']}")
                            st.text(f"Глава: {source['глава']}")
                            st.text(f"Текст: {source['текст'][:200]}...")
                            st.divider()
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"Ошибка при обработке запроса: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


# streamlit run app.py
# YjllY2FhYjgtNGRlMC00MDA4LWIwZmYtNjdlNjY0ZmI5OTc4OmI5YTY3NjYwLWJkMmQtNDNmZi04YzViLTU2MWMxYTE0MjFlMw==