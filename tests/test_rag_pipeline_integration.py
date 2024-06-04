# test_rag_pipeline_integration.py
import pytest
from rag_pipeline_ops import create_pipeline, run_pipeline
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
import os


load_dotenv()

@pytest.fixture
def sample_documents():
    return [
        "Artificial Intelligence is a branch of computer science.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural Language Processing enables machines to understand human language."
    ]

@pytest.fixture
def sample_query():
    return "What is machine learning?"

@patch('rag_pipeline_ops.OpenAIEmbedding')
@patch('rag_pipeline_ops.OpenAIQuery')
def test_rag_pipeline_with_hnswlib(MockOpenAIQuery, MockOpenAIEmbedding, sample_documents, sample_query):
    mock_embedding_instance = MockOpenAIEmbedding.return_value
    mock_query_instance = MockOpenAIQuery.return_value
    
    mock_embedding_instance.embed.return_value = [[0.1, 0.2, 0.3]] * len(sample_documents)
    mock_embedding_instance.get_dimension.return_value = 3
    mock_query_instance.generate.return_value = "This is a mock answer."
    
    api_key = os.getenv("OPENAI_API_KEY")
    pipeline = create_pipeline(
        embedding_model_name='openai',
        query_model_name='gpt-3',
        vector_db='hnswlib',
        api_key=api_key,
        similarity='cosine'
    )
    pipeline.add_documents(sample_documents)
    result = run_pipeline(pipeline, sample_documents, sample_query, k=3)
    assert result['query'] == sample_query
    assert len(result['relevant_docs']) == 3

@patch('rag_pipeline_ops.OpenAIEmbedding')
@patch('rag_pipeline_ops.OpenAIQuery')
def test_rag_pipeline_with_chromadb(MockOpenAIQuery, MockOpenAIEmbedding, sample_documents, sample_query):
    mock_embedding_instance = MockOpenAIEmbedding.return_value
    mock_query_instance = MockOpenAIQuery.return_value
    
    mock_embedding_instance.embed.return_value = [[0.1, 0.2, 0.3]] * len(sample_documents)
    mock_embedding_instance.get_dimension.return_value = 3
    mock_query_instance.generate.return_value = "This is a mock answer."
    
    api_key = os.getenv("OPENAI_API_KEY")
    pipeline = create_pipeline(
        embedding_model_name='openai',
        query_model_name='gpt-3',
        vector_db='chromadb',
        api_key=api_key,
        similarity='cosine',
        use_persistent_storage=True
    )
    pipeline.add_documents(sample_documents)
    result = run_pipeline(pipeline, sample_documents, sample_query, k=3)
    assert result['query'] == sample_query
    assert len(result['relevant_docs']) == 3