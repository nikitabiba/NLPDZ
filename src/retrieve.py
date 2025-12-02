from Retriever import Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings


JSON_PATH = r"C:\Users\My Computer\Desktop\Work\Learn\NLPDZ\data\json"
INDEX_NAME = "index"
VECTOR_STORE_PATH = r"C:\Users\My Computer\Desktop\Work\Learn\NLPDZ\data\vector_store"

emdedding_model = HuggingFaceEmbeddings(model_name="sberbank-ai/sbert_large_nlu_ru")
retriever = Retriever(emdedding_model=emdedding_model, json_data_path=JSON_PATH, index_file_name=INDEX_NAME, vector_store_path=VECTOR_STORE_PATH)

retriever.load_data()
results = retriever.retrieve(query="Когда могут быть отменены постановления правительства Российской Федерации?", k=10)

for doc in results:
    meta = doc.metadata

    print(f"Раздел: {meta.get('раздел', '')}")
    print(f"Глава: {meta.get('глава', '')}")
    print(f"Статья: {meta.get('статья', '')} | Пункт: {meta.get('пункт', '')}")
    print(f"Текст: {doc.page_content[:200]}...")
    print("-" * 60)

