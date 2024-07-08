import pytest
from unittest.mock import Mock, patch
import numpy as np
from bombay.pipeline.rag_pipeline import RAGPipeline
from bombay.pipeline.vector_db import HNSWLib, ChromaDB
from bombay.utils.config import Config

@pytest.fixture
def mock_config():
    mock_config = Mock(spec=Config)
    mock_config.some_config_value = 'mocked_value'
    return mock_config

@pytest.fixture
def mock_embedding():
    mock = Mock()
    mock.get_dimension.return_value = 3
    mock.embed.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    return mock

@pytest.fixture
def mock_query():
    mock = Mock()
    mock.generate.return_value = "This is a mock answer."
    return mock

def test_hnswlib_add_and_search(mock_embedding):
    hnswlib_db = HNSWLib(dim=mock_embedding.get_dimension())
    documents = ["doc1", "doc2", "doc3"]
    embeddings = mock_embedding.embed(documents)
    hnswlib_db.add_documents(documents, embeddings)
    query_embedding = mock_embedding.embed(["query"])[0]
    results = hnswlib_db.search(query_embedding, k=2)
    assert len(results) == 2

@patch('bombay.pipeline.vector_db.chromadb.Client')
def test_chromadb_add_and_search(mock_chromadb_client, mock_embedding):
    mock_collection = Mock()
    mock_chromadb_client.return_value.create_collection.return_value = mock_collection
    mock_collection.query.return_value = {
        'documents': [['doc1', 'doc2']],
        'distances': [[0.1, 0.2]]
    }
    
    chromadb_db = ChromaDB(collection_name='test_collection', use_persistent_storage=False)
    documents = ["doc1", "doc2", "doc3"]
    embeddings = mock_embedding.embed(documents)
    chromadb_db.add_documents(documents, embeddings)
    query_embedding = mock_embedding.embed(["query"])[0]
    results = chromadb_db.search(query_embedding, k=2)
    assert len(results) == 2

def test_rag_pipeline_initialization(mock_embedding, mock_query, mock_config):
    pipeline = RAGPipeline(embedding_model=mock_embedding, query_model=mock_query, vector_db='hnswlib', config=mock_config)
    assert isinstance(pipeline.vector_db, HNSWLib)

def test_rag_pipeline_add_documents(mock_embedding, mock_query, mock_config):
    pipeline = RAGPipeline(embedding_model=mock_embedding, query_model=mock_query, vector_db='hnswlib', config=mock_config)
    documents = ["doc1", "doc2", "doc3"]
    pipeline.add_documents(documents)
    assert mock_embedding.embed.called
    assert len(pipeline.vector_db.documents) == 3

def test_rag_pipeline_search_and_answer(mock_embedding, mock_query, mock_config):
    pipeline = RAGPipeline(embedding_model=mock_embedding, query_model=mock_query, vector_db='hnswlib', config=mock_config)
    documents = ["doc1", "doc2", "doc3"]
    mock_embedding.embed.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    pipeline.add_documents(documents)
    mock_embedding.embed.return_value = np.array([[0.1, 0.2, 0.3]])
    result = pipeline.search_and_answer("query")
    assert 'query' in result
    assert 'relevant_docs' in result
    assert 'distances' in result
    assert 'answer' in result
    assert result['answer'] == "This is a mock answer."