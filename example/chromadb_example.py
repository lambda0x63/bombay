#chromadb_example.py
from bombay.pipeline import create_pipeline, run_pipeline
from dotenv import load_dotenv
import os

load_dotenv()

# OpenAI API 키 설정
api_key = os.getenv("OPENAI_API_KEY")

# Bombay 파이프라인 생성 (온메모리)
pipeline_inmemory = create_pipeline(
    embedding_model_name='openai',
    query_model_name='gpt-3',
    vector_db='chromadb',
    api_key=api_key,
    similarity='cosine',
    use_persistent_storage=False
)

# Bombay 파이프라인 생성 (영구 저장)
pipeline_persistent = run_pipeline(
    embedding_model_name='openai',
    query_model_name='gpt-3',
    vector_db='chromadb',
    api_key=api_key,
    similarity='cosine',
    use_persistent_storage=True
)

# 문서 추가 (온메모리)
documents = [
    "고양이는 앞발에 5개, 뒷발에 4개의 발가락이 있습니다.",
    "고양이는 수면 시간이 많아 하루 평균 15~20시간을 잡니다.",
    "고양이는 점프력이 뛰어나 자신의 몸길이의 최대 6배까지 뛰어오를 수 있습니다."
]
pipeline_inmemory.add_documents(documents)

# 문서 추가 (영구 저장)
pipeline_persistent.add_documents(documents)

# 검색 및 질의 응답 (온메모리)
query = "고양이의 수면 시간은 어떻게 되나요?"
result_inmemory = run_pipeline(pipeline_inmemory, documents, query, k=1)

# 검색 및 질의 응답 (영구 저장)
result_persistent = run_pipeline(pipeline_persistent, documents, query, k=1)

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