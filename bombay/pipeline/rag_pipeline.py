# bombay/pipeline/rag_pipeline.py

import numpy as np
from .vector_db import VectorDB, HNSWLib, ChromaDB
from .embedding_models import EmbeddingModel, OpenAIEmbedding
from .query_models import QueryModel, OpenAIQuery
from ..utils.logging import logger
from ..utils.preprocessing import preprocess_text

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