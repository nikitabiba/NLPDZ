import os
import json
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Vectorizer:
    def __init__(self, embedding_model, path, index_file_name, vector_store_path):
        self.embedding_model = embedding_model
        self.path = path
        self.index_file_name = index_file_name
        self.vector_store_path = vector_store_path
        self.docs = []
        self.faiss_index = None

    def vectorize(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Директория {self.path} не найдена")
        
        self.files = [f for f in os.listdir(self.path)]

        if all(f.endswith(".json") for f in self.files):
            self.vectorize_json()
        else:
            raise ValueError("В директории присутствуют файлы с неизвестным форматом")

    def vectorize_json(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
        )

        for filename in tqdm(self.files, desc="Обработка файлов"):
            file_path = os.path.join(self.path, filename)
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Ошибка чтения файла {filename}: {e}")
                continue
            except Exception as e:
                print(f"Неизвестная ошибка при чтении файла {filename}: {e}")
                continue

            for section_name, chapters in data.items():
                for chapter_name, articles in chapters.items():
                    for article_name, points in articles.items():
                        for point_name, text in points.items():
                            text = text.strip()
                            if not text:
                                continue
                            chunks = splitter.split_text(text)

                            metadata = {
                                "раздел": section_name,
                                "глава": chapter_name,
                                "статья": article_name,
                                "пункт": point_name
                            }

                            for chunk in chunks:
                                self.docs.append(Document(page_content=chunk, metadata=metadata))

        print(f"Векторизация {len(self.docs)} фрагментов...")
        
        try:
            self.faiss_index = FAISS.from_documents(self.docs, self.embedding_model)
            print("Векторизация завершена.")
        except Exception as e:
            raise RuntimeError(f"Ошибка при создании FAISS индекса: {e}")

    def save(self):
        if self.faiss_index is None:
            raise RuntimeError("Индекс не создан. Сначала выполните векторизацию.")
        
        try:
            os.makedirs(self.vector_store_path, exist_ok=True)
            
            self.faiss_index.save_local(
                folder_path=self.vector_store_path, 
                index_name=self.index_file_name
            )
            
            print(f"\nВекторный индекс и метаданные сохранены в: {self.vector_store_path}")
            print(f"  - {self.index_file_name}.faiss")
            print(f"  - {self.index_file_name}.pkl")
            
        except Exception as e:
            raise RuntimeError(f"Ошибка при сохранении индекса: {e}")
    
    def get_document_count(self):
        return len(self.docs)

