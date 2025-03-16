# Bombay Pipeline

Bombay는 RAG(Retrieval-Augmented Generation) 기반 LLM(Large Language Model) 구축 및 활용을 위한 파이프라인 시스템입니다.
Python 3.12 이상 버전에서 안정적 동작을 보장하며, 타 버전 호환성은 추후 단위테스트 예정입니다.

## 주요 기능

- **다양한 모델 지원**: OpenAI Embedding 모델과 GPT 모델 지원. 추후 모델 확장 예정
- **벡터 데이터베이스 통합**: Hnswlib, ChromaDB 지원. 추후 온프레미스 및 클라우드 환경 확장 예정
- **문서 관리**: 통합 인터페이스를 통한 문서 CRUD 기능 제공 (테스트 진행 중)

## 설치

```bash
pip install bombay
```

## 사용 방법

### 프로젝트 생성

```bash
bombay
```

#### 지원 템플릿
- Basic: 기본 파이프라인 구성
- Chatbot: 대화형 단일 채팅 기능 포함
- Web App: FastAPI 기반 웹 애플리케이션

#### 프로젝트 구조
```
<project_name>/
├── main.py
└── .env
```

### 파이프라인 구성

```python
from bombay.pipeline import create_pipeline

pipeline = create_pipeline(
    embedding_model_name='openai',
    query_model_name='gpt-3',
    vector_db='chromadb',
    api_key='YOUR_API_KEY',
    similarity='cosine',
    use_persistent_storage=True
)
```

#### 매개변수
- `embedding_model_name`: 임베딩 모델명 (현재 'openai' 지원)
- `query_model_name`: 질의 모델명 (현재 'gpt-3' 지원)
- `vector_db`: 벡터 데이터베이스 ('hnswlib' 또는 'chromadb')
- `api_key`: OpenAI API 키
- `similarity`: 유사도 측정 방식 (기본값: 'cosine')
- `use_persistent_storage`: 데이터 지속성 여부 (기본값: False)

### 문서 추가

```python
documents = [
    "고양이는 포유류에 속하는 동물입니다.",
    "고양이는 약 6,000년 전부터 인간과 함께 살아온 것으로 추정됩니다.",
    "고양이는 예민한 청각과 후각을 가지고 있어 작은 움직임이나 냄새도 쉽게 감지할 수 있습니다.",
    "고양이는 앞발에 5개, 뒷발에 4개의 발가락이 있습니다.",
    "고양이는 수면 시간이 많아 하루 평균 15~20시간을 잡니다.",
    "고양이는 점프력이 뛰어나 자신의 몸길이의 최대 6배까지 뛰어오를 수 있습니다."
]

pipeline.add_documents(documents)
```

### 검색 및 응답 생성

```python
query = "고양이는 어떤 동물인가요?"
result = run_pipeline(pipeline, documents, query, k=2)

print(f"질문: {result['query']}")
print(f"관련 문서: {result['relevant_docs']}")
print(f"답변: {result['answer']}")
```

#### 실행 결과
```
질문: 고양이는 어떤 동물인가요?
관련 문서: ['고양이는 포유류에 속하는 동물입니다.', '고양이는 약 6,000년 전부터 인간과 함께 살아온 것으로 추정됩니다.']
답변: 고양이는 포유류에 속하는 동물로, 약 6,000년 전부터 인간과 함께 살아온 것으로 추정됩니다.
```

## 설계 원칙

- **추상화와 인터페이스**: 벡터 데이터베이스, 임베딩 모델, 질의 모델에 대한 추상 클래스 정의
- **팩토리 패턴**: `create_pipeline` 함수를 통한 파이프라인 구성요소 생성
- **어댑터 패턴**: OpenAI API를 추상화된 인터페이스에 맞게 적용
