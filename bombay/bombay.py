#bombay.py
import platform
import numpy as np
import hnswlib
import chromadb
from chromadb.config import Settings
from uuid import uuid4
import os
from openai import OpenAI
from abc import ABC, abstractmethod

# 벡터 DB 추상 클래스
class VectorDB(ABC):
    @abstractmethod
    def __init__(self):
        self.documents = []

    @abstractmethod
    def add_documents(self, documents, embeddings):
        """
        문서와 임베딩을 벡터 DB에 추가하는 메소드
        :param documents: 추가할 문서 리스트
        :param embeddings: 문서에 해당하는 임베딩 리스트
        """
        pass

    @abstractmethod
    def update_document(self, document_id, document, embedding):
        """
        문서를 업데이트하는 메소드
        :param document_id: 업데이트할 문서의 ID
        :param document: 새로운 문서
        :param embedding: 새로운 문서의 임베딩
        """
        pass

    @abstractmethod
    def delete_document(self, document_id):
        """
        문서를 삭제하는 메소드
        :param document_id: 삭제할 문서의 ID
        """
        pass

    @abstractmethod
    def search(self, query_embedding, k=1, threshold=None):
        """
        쿼리 임베딩과 유사한 문서를 검색하는 메소드
        :param query_embedding: 쿼리의 임베딩
        :param k: 검색할 문서의 개수 (기본값: 1)
        :param threshold: 유사도 임계값 (기본값: None)
        :return: (문서, 유사도) 튜플의 리스트
        """
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

# 임베딩 모델 추상 클래스
class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, texts):
        """
        텍스트를 임베딩하는 메소드
        :param texts: 임베딩할 텍스트 리스트
        :return: 임베딩 리스트
        """
        pass

    @abstractmethod
    def get_dimension(self):
        """
        임베딩의 차원을 반환하는 메소드
        :return: 임베딩의 차원
        """
        pass

# OpenAI 임베딩 모델 어댑터
class OpenAIEmbedding(EmbeddingModel):
    def __init__(self, api_key, model):
        """
        OpenAI 임베딩 모델 초기화
        :param api_key: OpenAI API 키
        :param model: 사용할 OpenAI 임베딩 모델
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimension = None

    def embed(self, texts):
        """
        텍스트를 OpenAI 임베딩 모델로 임베딩하는 메소드
        :param texts: 임베딩할 텍스트 리스트
        :return: 임베딩 리스트
        """
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        embeddings = [data.embedding for data in response.data]
        if self.dimension is None:
            self.dimension = len(embeddings[0])
        return embeddings

    def get_dimension(self):
        """
        OpenAI 임베딩 모델의 임베딩 차원을 반환하는 메소드
        :return: 임베딩의 차원
        """
        if self.dimension is None:
            sample_document = 'This is a sample document to get embedding dimension.'
            self.dimension = len(self.embed([sample_document])[0])
        return self.dimension

# RAG 파이프라인 클래스
class RAGPipeline:
    def __init__(self, embedding_model, query_model, vector_db, similarity='cosine', **kwargs):
        """
        RAG 파이프라인 초기화
        :param embedding_model: 임베딩 모델
        :param query_model: 질의 모델
        :param vector_db: 벡터 DB 이름 또는 인스턴스
        :param similarity: 유사도 측정 방식 (기본값: 'cosine')
        :param **kwargs: 벡터 DB 초기화에 사용되는 추가 인자
        """
        self.embedding_model = embedding_model
        self.query_model = query_model
        self.similarity = similarity
        self.vector_db = self._initialize_vector_db(vector_db, **kwargs)
        
    def _initialize_vector_db(self, vector_db, **kwargs):
        """
        벡터 DB를 초기화하는 메소드
        :param vector_db: 벡터 DB 이름 또는 인스턴스
        :param **kwargs: 벡터 DB 초기화에 사용되는 추가 인자
        :return: 초기화된 벡터 DB 인스턴스
        """
        if isinstance(vector_db, str):
            if vector_db.lower() == 'hnswlib':
                return HNSWLib(self.embedding_model.get_dimension(), similarity=self.similarity)
            elif vector_db.lower() == 'chromadb':
                return ChromaDB(**kwargs)
            else:
                raise ValueError(f"Unsupported vector database: {vector_db}")
        elif isinstance(vector_db, VectorDB):
            return vector_db
        else:
            raise ValueError(f"Unsupported vector database type: {type(vector_db)}")

    def add_documents(self, documents):
        """
        문서를 RAG 파이프라인에 추가하는 메소드
        :param documents: 추가할 문서 리스트
        """
        embeddings = self.embedding_model.embed(documents)
        self.vector_db.add_documents(documents, np.array(embeddings))

    def update_document(self, document_id, document):
        """
        문서를 업데이트하는 메소드
        :param document_id: 업데이트할 문서의 ID
        :param document: 새로운 문서
        """
        embedding = self.embedding_model.embed([document])[0]
        self.vector_db.update_document(document_id, document, embedding)

    def delete_document(self, document_id):
        """
        문서를 삭제하는 메소드
        :param document_id: 삭제할 문서의 ID
        """
        self.vector_db.delete_document(document_id)

    def search_and_answer(self, query, k=1, threshold=None):
        """
        쿼리를 검색하고 관련 문서를 사용하여 답변을 생성하는 메소드
        :param query: 검색할 쿼리
        :param k: 검색할 문서의 개수 (기본값: 1)
        :param threshold: 유사도 임계값 (기본값: None)
        :return: (쿼리, 관련 문서, 유사도, 답변)
        튜플
        """

# 질의 모델 추상 클래스
class QueryModel(ABC):
    @abstractmethod
    def generate(self, query, relevant_docs):
        """
        쿼리와 관련 문서를 사용하여 답변을 생성하는 메소드
        :param query: 사용자 쿼리
        :param relevant_docs: 관련 문서 리스트
        :return: 생성된 답변
        """
        pass

# GPT 기반 질의 모델 어댑터
class OpenAIQuery(QueryModel):
    def __init__(self, api_key, model):
        """
        GPT 기반 질의 모델 초기화
        :param api_key: OpenAI API 키
        :param model: 사용할 GPT 모델
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, query, relevant_docs):
        """
        쿼리와 관련 문서를 사용하여 GPT로 답변을 생성하는 메소드
        :param query: 사용자 쿼리
        :param relevant_docs: 관련 문서 리스트
        :return: 생성된 답변
        """
        relevant_docs_str = ' '.join(relevant_docs)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"Be sure to refer to Relevant documents to answer questions. Relevant documents: {relevant_docs_str} "},
                {"role": "user", "content": f"questions: {query}"}
            ]
        )
        return response.choices[0].message.content

# RAG 파이프라인 생성 함수
def create_pipeline(embedding_model_name, query_model_name, vector_db, api_key, similarity='cosine', use_persistent_storage=False, **kwargs):
    """
    RAG 파이프라인을 생성하는 함수
    :param embedding_model_name: 임베딩 모델 이름
    :param query_model_name: 질의 모델 이름
    :param vector_db: 벡터 DB 이름 또는 인스턴스
    :param api_key: OpenAI API 키
    :param similarity: 유사도 측정 방식 (기본값: 'cosine')
    :param use_persistent_storage: 영구 저장소 사용 여부 (기본값: False)
    :param **kwargs: 벡터 DB 초기화에 사용되는 추가 인자
    :return: 생성된 RAG 파이프라인
    """
    embedding_models = {
        'openai': OpenAIEmbedding(api_key, 'text-embedding-ada-002')
    }
    query_models = {
        'gpt-3': OpenAIQuery(api_key, 'gpt-3.5-turbo')
    }

    embedding_model = embedding_models.get(embedding_model_name)
    query_model = query_models.get(query_model_name)

    if embedding_model is None:
        raise ValueError(f"Unsupported embedding model: {embedding_model_name}")
    if query_model is None:
        raise ValueError(f"Unsupported query model: {query_model_name}")

    if isinstance(vector_db, str) and vector_db.lower() == 'chromadb':
        return RAGPipeline(embedding_model, query_model, vector_db, similarity, use_persistent_storage=use_persistent_storage, **kwargs)
    else:
        return RAGPipeline(embedding_model, query_model, vector_db, similarity, **kwargs)

# RAG 파이프라인 실행 함수
def run_pipeline(pipeline, documents, query, k=1, threshold=None):
    """
    RAG 파이프라인을 실행하는 함수
    :param pipeline: RAG 파이프라인 인스턴스
    :param documents: 검색할 문서 리스트
    :param query: 사용자 쿼리
    :param k: 검색할 문서의 개수 (기본값: 1)
    :param threshold: 유사도 임계값 (기본값: None)
    :return: 검색 결과 (쿼리, 관련 문서, 유사도, 답변)
    """
    query_embedding = pipeline.embedding_model.embed([query])[0]
    relevant_docs, distances = zip(*pipeline.vector_db.search(query_embedding, k, threshold))
    answer = pipeline.query_model.generate(query, relevant_docs)
    return {
        'query': query,
        'relevant_docs': relevant_docs,
        'distances': distances,
        'answer': answer
    }