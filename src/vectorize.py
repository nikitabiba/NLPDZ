from Vectorizer import Vectorizer
from langchain_community.embeddings import HuggingFaceEmbeddings


JSON_PATH = r"C:\Users\My Computer\Desktop\Work\Learn\NLPDZ\data\json"
INDEX_NAME = "index"
VECTOR_STORE_PATH = r"C:\Users\My Computer\Desktop\Work\Learn\NLPDZ\data\vector_store"

emdedding_model = HuggingFaceEmbeddings(model_name="sberbank-ai/sbert_large_nlu_ru")
vectorizer = Vectorizer(embedding_model=emdedding_model, path=JSON_PATH, index_file_name=INDEX_NAME, vector_store_path=VECTOR_STORE_PATH)

vectorizer.vectorize()
vectorizer.save()
print(vectorizer.get_document_count())

