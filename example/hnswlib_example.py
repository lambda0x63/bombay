from rag_pipeline_ops import create_rag_pipeline, run_rag_pipeline
from dotenv import load_dotenv
import os

load_dotenv()
# OpenAI API 키 설정
api_key = os.getenv("OPENAI_API_KEY")

# RAG 파이프라인 생성
pipeline = create_rag_pipeline(
    embedding_model_name='openai',
    query_model_name='gpt-3',
    vector_db='hnswlib',
    api_key=api_key,
    similarity='cosine'
)

# 문서 추가
documents = [
    "Artificial Intelligence is a branch of computer science.",
    "Machine learning is a subset of artificial intelligence.",
    "Natural Language Processing enables machines to understand human language."
]
pipeline.add_documents(documents)

# 검색 및 질의 응답
query = "What is machine learning?"
result = run_rag_pipeline(pipeline, documents, query, k=2)

# 결과 출력
print(f"Query: {result['query']}")
print(f"Relevant Documents: {result['relevant_docs']}")
print(f"Distances: {result['distances']}")
print(f"Answer: {result['answer']}")