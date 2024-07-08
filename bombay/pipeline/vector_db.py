#bombay/pipeline/vector_db.py
from abc import ABC, abstractmethod
import numpy as np
import hnswlib
import chromadb
from chromadb.config import Settings
from uuid import uuid4
import os

class VectorDB(ABC):
    @abstractmethod
    def __init__(self):
        self.documents = []

    @abstractmethod
    def add_documents(self, documents, embeddings):
        pass

    @abstractmethod
    def update_document(self, document_id, document, embedding):
        pass

    @abstractmethod
    def delete_document(self, document_id):
        pass

    @abstractmethod
    def search(self, query_embedding, k=1, threshold=None):
        pass

# Hnswlib 벡터 DB 어댑터
class HNSWLib(VectorDB):
    def __init__(self, dim, similarity='cosine'):
        """
        Hnswlib 벡터 DB 초기화
        :param dim: 벡터의 차원
        :param similarity: 유사도 측정 방식 (기본값: 'cosine')
        """
        super().__init__()
        self.index = hnswlib.Index(space=similarity, dim=dim)
        self.similarity = similarity
        self.document_ids = []

    def add_documents(self, documents, embeddings):
        """
        문서와 임베딩을 Hnswlib 벡터 DB에 추가하는 메소드
        :param documents: 추가할 문서 리스트
        :param embeddings: 문서에 해당하는 임베딩 리스트
        """
        if self.index.max_elements == 0:
            self.index.init_index(max_elements=len(documents), ef_construction=2000, M=64)
        elif self.index.element_count + len(documents) > self.index.max_elements:
            self.index.resize_index(self.index.element_count + len(documents))

        embeddings = np.float32(embeddings)
        ids = np.arange(len(self.document_ids), len(self.document_ids) + len(documents))

        self.documents.extend(documents)
        self.document_ids.extend(ids)
        self.index.add_items(embeddings, ids)

    def update_document(self, document_id, document, embedding):
        """
        문서를 업데이트하는 메소드
        :param document_id: 업데이트할 문서의 ID
        :param document: 새로운 문서
        :param embedding: 새로운 문서의 임베딩
        """
        if document_id in self.document_ids:
            index = self.document_ids.index(document_id)
            self.documents[index] = document
            self.index.mark_deleted(index)
            self.index.add_items([embedding], [index])
        else:
            raise ValueError(f"Document with id {document_id} not found.")

    def delete_document(self, document_id):
        """
        문서를 삭제하는 메소드
        :param document_id: 삭제할 문서의 ID
        """
        if document_id in self.document_ids:
            index = self.document_ids.index(document_id)
            del self.documents[index]
            del self.document_ids[index]
            self.index.mark_deleted(index)
        else:
            raise ValueError(f"Document with id {document_id} not found.")

    def search(self, query_embedding, k=1, threshold=None):
        """
        쿼리 임베딩과 유사한 문서를 Hnswlib 벡터 DB에서 검색하는 메소드
        :param query_embedding: 쿼리의 임베딩
        :param k: 검색할 문서의 개수 (기본값: 1)
        :param threshold: 유사도 임계값 (기본값: None)
        :return: (문서, 유사도) 튜플의 리스트
        """
        indices, distances = self.index.knn_query([query_embedding], k=k)
        indices = indices[0]
        distances = distances[0]
        if threshold is not None:
            mask = distances <= threshold
            indices = indices[mask]
            distances = distances[mask]
        return [(self.documents[self.document_ids[idx]], dist) for idx, dist in zip(indices, distances)]

# ChromaDB 클래스
class ChromaDB(VectorDB):
    def __init__(self, collection_name='default', use_persistent_storage=False, embedding_function=None):
        """
        ChromaDB 초기화
        :param collection_name: 컬렉션 이름 (기본값: 'default')
        :param use_persistent_storage: 영구 저장소 사용 여부 (기본값: False)
        :param embedding_function: 임베딩 함수 (기본값: None)
        """
        super().__init__()
        self.persist_directory = './chromadb_persist' if use_persistent_storage else None
        self.embedding_function = embedding_function
        if self.persist_directory:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
        else:
            self.client = chromadb.Client()
        
        try:
            self.collection = self.client.create_collection(name=collection_name, embedding_function=embedding_function)
        except chromadb.db.base.UniqueConstraintError:
            print(f"Collection '{collection_name}' already exists. Using existing collection.")
            self.collection = self.client.get_collection(name=collection_name, embedding_function=embedding_function)

    def add_documents(self, documents, embeddings, metadatas=None, ids=None):
        """
        문서와 임베딩을 ChromaDB에 추가하는 메소드
        :param documents: 추가할 문서 리스트
        :param embeddings: 문서에 해당하는 임베딩 리스트
        :param metadatas: 문서 메타데이터 리스트 (기본값: None)
        :param ids: 문서 ID 리스트 (기본값: None)
        """
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(documents))]
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def update_document(self, document_id, document=None, embedding=None, metadata=None):
        """
        문서를 업데이트하는 메소드
        :param document_id: 업데이트할 문서의 ID
        :param document: 새로운 문서 (기본값: None)
        :param embedding: 새로운 문서의 임베딩 (기본값: None)
        :param metadata: 새로운 문서의 메타데이터 (기본값: None)
        """
        self.collection.update(
            ids=[document_id],
            documents=[document] if document else None,
            embeddings=[embedding] if embedding else None,
            metadatas=[metadata] if metadata else None
        )

    def delete_document(self, document_id):
        """
        문서를 삭제하는 메소드
        :param document_id: 삭제할 문서의 ID
        """
        self.collection.delete(ids=[document_id])

    def search(self, query_embedding, k=1, threshold=None, where=None):
        """
        쿼리 임베딩과 유사한 문서를 ChromaDB에서 검색하는 메소드
        :param query_embedding: 쿼리의 임베딩
        :param k: 검색할 문서의 개수 (기본값: 1)
        :param threshold: 유사도 임계값 (기본값: None)
        :param where: 검색 조건 (기본값: None)
        :return: (문서, 유사도) 튜플의 리스트
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where
        )
        distances = results['distances'][0][:k]
        documents = results['documents'][0][:k]
        return list(zip(documents, distances))