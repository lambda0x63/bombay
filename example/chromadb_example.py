from rag_pipeline_ops import create_rag_pipeline, run_rag_pipeline
from dotenv import load_dotenv
import os

load_dotenv()

# OpenAI API 키 설정
api_key = os.getenv("OPENAI_API_KEY")

# RAG 파이프라인 생성 (온메모리)
pipeline_inmemory = create_rag_pipeline(
    embedding_model_name='openai',
    query_model_name='gpt-3',
    vector_db='chromadb',
    api_key=api_key,
    similarity='cosine',
    use_persistent_storage=False
)

# RAG 파이프라인 생성 (영구 저장)
pipeline_persistent = create_rag_pipeline(
    embedding_model_name='openai',
    query_model_name='gpt-3',
    vector_db='chromadb',
    api_key=api_key,
    similarity='cosine',
    use_persistent_storage=True
)

# 문서 추가 (온메모리)
documents = [
    "Artificial Intelligence is a branch of computer science.",
    "Machine learning is a subset of artificial intelligence.",
    "Natural Language Processing enables machines to understand human language."
]
pipeline_inmemory.add_documents(documents)

# 문서 추가 (영구 저장)
pipeline_persistent.add_documents(documents)

# 검색 및 질의 응답 (온메모리)
query = "What is machine learning?"
result_inmemory = run_rag_pipeline(pipeline_inmemory, documents, query, k=2)

# 검색 및 질의 응답 (영구 저장)
result_persistent = run_rag_pipeline(pipeline_persistent, documents, query, k=2)

# 결과 출력 (온메모리)
print("In-Memory Results:")
print(f"Query: {result_inmemory['query']}")
print(f"Relevant Documents: {result_inmemory['relevant_docs']}")
print(f"Distances: {result_inmemory['distances']}")
print(f"Answer: {result_inmemory['answer']}")

# 결과 출력 (영구 저장)
print("\nPersistent Storage Results:")
print(f"Query: {result_persistent['query']}")
print(f"Relevant Documents: {result_persistent['relevant_docs']}")
print(f"Distances: {result_persistent['distances']}")
print(f"Answer: {result_persistent['answer']}")