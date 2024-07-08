import pytest
from unittest.mock import patch, MagicMock
from bombay import create_pipeline, run_pipeline
import numpy as np

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

@patch('bombay.pipeline.rag_pipeline.OpenAIEmbedding')
@patch('bombay.pipeline.rag_pipeline.OpenAIQuery')
def test_rag_pipeline_with_hnswlib(MockOpenAIQuery, MockOpenAIEmbedding, sample_documents, sample_query):
    mock_embedding_instance = MockOpenAIEmbedding.return_value
    mock_query_instance = MockOpenAIQuery.return_value
    
    mock_embedding_instance.embed.return_value = np.array([[0.1, 0.2, 0.3]] * len(sample_documents))
    mock_embedding_instance.get_dimension.return_value = 3
    mock_query_instance.generate.return_value = "This is a mock answer."
    
    pipeline = create_pipeline(
        embedding_model_name='openai',
        query_model_name='gpt-3',
        vector_db='hnswlib',
        api_key='dummy_api_key_for_testing',
        similarity='cosine'
    )
    pipeline.add_documents(sample_documents)
    result = run_pipeline(pipeline, sample_documents, sample_query, k=3)
    assert result['query'] == sample_query
    assert len(result['relevant_docs']) == 3
    assert result['answer'] == "This is a mock answer."

@patch('bombay.pipeline.rag_pipeline.OpenAIEmbedding')
@patch('bombay.pipeline.rag_pipeline.OpenAIQuery')
def test_rag_pipeline_with_chromadb(MockOpenAIQuery, MockOpenAIEmbedding, sample_documents, sample_query):
    mock_embedding_instance = MockOpenAIEmbedding.return_value
    mock_query_instance = MockOpenAIQuery.return_value
    
    mock_embedding_instance.embed.return_value = np.array([[0.1, 0.2, 0.3]] * len(sample_documents))
    mock_embedding_instance.get_dimension.return_value = 3
    mock_query_instance.generate.return_value = "This is a mock answer."
    
    pipeline = create_pipeline(
        embedding_model_name='openai',
        query_model_name='gpt-3',
        vector_db='chromadb',
        api_key='dummy_api_key_for_testing',
        similarity='cosine',
        use_persistent_storage=False
    )
    pipeline.add_documents(sample_documents)
    result = run_pipeline(pipeline, sample_documents, sample_query, k=3)
    assert result['query'] == sample_query
    assert len(result['relevant_docs']) == 3
    assert result['answer'] == "This is a mock answer."