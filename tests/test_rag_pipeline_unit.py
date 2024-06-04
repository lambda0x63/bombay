#test_rag_pipeline_unit.py
import pytest
import os
from rag_pipeline_ops import RAGPipeline, HNSWLib, ChromaDB
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Mock classes
class MockEmbedding:
    def __init__(self):
        self.dimension = 3

    def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def get_dimension(self):
        return self.dimension

class MockQuery:
    def generate(self, query, relevant_docs):
        return "This is a mock answer."

@pytest.fixture
def mock_embedding():
    return MockEmbedding()

@pytest.fixture
def mock_query():
    return MockQuery()

def test_hnswlib_add_and_search(mock_embedding):
    hnswlib_db = HNSWLib(dim=mock_embedding.get_dimension())
    documents = ["doc1", "doc2", "doc3"]
    embeddings = mock_embedding.embed(documents)
    hnswlib_db.add_documents(documents, embeddings)
    query_embedding = mock_embedding.embed(["query"])[0]
    results = hnswlib_db.search(query_embedding, k=2)
    assert len(results) == 2

def test_chromadb_add_and_search(mock_embedding, tmpdir):
    chromadb_test_data = tmpdir.mkdir("chromadb_test_data")
    chromadb_db = ChromaDB(collection_name='test_collection', use_persistent_storage=True, embedding_function=None)
    documents = ["doc1", "doc2", "doc3"]
    embeddings = mock_embedding.embed(documents)
    chromadb_db.add_documents(documents, embeddings)
    query_embedding = mock_embedding.embed(["query"])[0]
    results = chromadb_db.search(query_embedding, k=2)
    assert len(results) == 2

def test_rag_pipeline_initialization(mock_embedding, mock_query):
    pipeline = RAGPipeline(embedding_model=mock_embedding, query_model=mock_query, vector_db='hnswlib')
    assert isinstance(pipeline.vector_db, HNSWLib)

def test_rag_pipeline_add_documents(mock_embedding, mock_query):
    pipeline = RAGPipeline(embedding_model=mock_embedding, query_model=mock_query, vector_db='hnswlib')
    documents = ["doc1", "doc2", "doc3"]
    pipeline.add_documents(documents)
    assert len(pipeline.vector_db.documents) == 3

def test_rag_pipeline_search_and_answer(mock_embedding, mock_query):
    pipeline = RAGPipeline(embedding_model=mock_embedding, query_model=mock_query, vector_db='hnswlib')
    documents = ["doc1", "doc2", "doc3"]
    pipeline.add_documents(documents)
    query, relevant_docs, distances, answer = pipeline.search_and_answer("query")
    assert len(relevant_docs) == 1
    assert answer == "This is a mock answer."