import os
import json
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document


class Retriever:
    def __init__(self, emdedding_model, json_data_path, index_file_name, vector_store_path):
        self.emdedding_model = emdedding_model
        self.json_data_path = json_data_path
        self.index_file_name = index_file_name
        self.vector_store_path = vector_store_path
        
        self.faiss_index = None
        self.bm25_retriever = None
        self.faiss_retriever = None
        self.ensemble_retriever = None
        self.fuser = None
        self.reranker_model = None
        self.docs = []

    def load_data(self):
        try:
            self.faiss_index = FAISS.load_local(
                self.vector_store_path, 
                self.emdedding_model, 
                index_name=self.index_file_name, 
                allow_dangerous_deserialization=True
            )
            
            bm25_documents = self._load_documents_from_json()
            
            self.bm25_retriever = BM25Retriever.from_documents(bm25_documents)
            self.docs = bm25_documents
            
            print(f"Загружено {len(bm25_documents)} документов")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Ошибка загрузки данных: {e}")
        except Exception as e:
            raise RuntimeError(f"Неожиданная ошибка при загрузке данных: {e}")

    def _load_documents_from_json(self):
        if not os.path.exists(self.json_data_path):
            raise FileNotFoundError(f"Директория {self.json_data_path} не найдена")
        
        documents = []
        json_files = [f for f in os.listdir(self.json_data_path) if f.endswith(".json")]
        
        for filename in tqdm(json_files, desc="Загрузка JSON документов"):
            file_path = os.path.join(self.json_data_path, filename)
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"Ошибка чтения файла {filename}: {e}")
                continue
            except Exception as e:
                print(f"Неожиданная ошибка при чтении файла {filename}: {e}")
                continue

            for section_name, chapters in data.items():
                for chapter_name, articles in chapters.items():
                    for article_name, points in articles.items():
                        for point_name, text in points.items():
                            text = text.strip()
                            if not text:
                                continue

                            documents.append(Document(
                                page_content=text,
                                metadata={
                                    "раздел": section_name,
                                    "глава": chapter_name,
                                    "статья": article_name,
                                    "пункт": point_name
                                }
                            ))

        return documents

    def retrieve(self, query, k = 10):
        try:
            self.bm25_retriever.k = k
            self.faiss_retriever = self.faiss_index.as_retriever(search_kwargs={"k": k})
            
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.faiss_retriever],
                weights=[0.5, 0.5]
            )

            fused_docs = self.ensemble_retriever.get_relevant_documents(query)

            self.reranker_model = HuggingFaceCrossEncoder(model_name="qilowoq/bge-reranker-v2-m3-en-ru")

            pairs = [(query, doc.page_content) for doc in fused_docs]
            scores = self.reranker_model.score(pairs)

            reranked_docs = [
                doc for _, doc in sorted(
                    zip(scores, fused_docs),
                    key=lambda x: x[0],
                    reverse=True
                )
            ]

            return reranked_docs[:k]
            
        except Exception as e:
            raise RuntimeError(f"Ошибка при выполнении поиска: {e}")
    
