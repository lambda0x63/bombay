# hnswlib_example.py
from bombay.pipeline import create_pipeline, run_pipeline
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

pipeline = create_pipeline(
   embedding_model_name='openai',
   query_model_name='gpt-3',
   vector_db='hnswlib',
   api_key=api_key,
   similarity='cosine'
)

# 문서 추가
documents = [
   "고양이는 포유류에 속하는 동물입니다.",
   "고양이는 약 6,000년 전부터 인간과 함께 살아온 것으로 추정됩니다.",
   "고양이는 예민한 청각과 후각을 가지고 있어 작은 움직임이나 냄새도 쉽게 감지할 수 있습니다."
]
pipeline.add_documents(documents)

# 검색 및 질의 응답
query = "고양이는 어떤 동물인가요?"
result = run_pipeline(pipeline, documents, query, k=2)

# 결과 출력
print(f"Query: {result['query']}")
print(f"Relevant Documents: {result['relevant_docs']}")
print(f"Distances: {result['distances']}")
print(f"Answer: {result['answer']}")