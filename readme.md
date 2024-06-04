# Bombay Pipeline

Bombay는 RAG (Retrieval-Augmented Generation) 기반의 LLM (Large Language Model)을 쉽게 구축하고 활용할 수 있도록 하는 파이프라인 구축 시스템입니다. 
현재 Python 3.12 이상 버전에서 Stable 합니다. 다른 버전에 대해서는 추후 단위테스트를 할 예정입니다.

## 주요 기능

- **다양한 모델 지원**: 다양한 임베딩 모델과 질의 모델을 지원하여 유연성을 제공합니다. 현재는 OpenAI의 Embedding 모델과 GPT 모델을 지원하며, 추후 더 많은 모델이 추가될 예정입니다.
- **벡터 데이터베이스 통합**: Hnswlib과 ChromaDB와 같은 고성능 벡터 데이터베이스를 추상 클래스를 통해 쉽게 세팅할 수 있도록 지원합니다. 추후 온프레미스 뿐만 아니라 클라우드 환경의 벡터 데이터베이스도 지원할 예정입니다.
- **문서 관리**: 문서 추가, 업데이트, 삭제 인터페이스를 통합된 인터페이스로 제공합니다. 해당 CRUD 기능은 테스트 진행 중입니다.

## 설치 및 운영환경

1. 해당 라이브러리를 `pip`으로 설치:
~~~bash
$ pip install bombay
~~~

## 사용 방법

### 1. Bombay 프로젝트 생성

Bombay CLI를 사용하여 새 프로젝트를 생성할 수 있습니다. 다음 명령어를 사용하여 CLI를 실행합니다:

~~~bash
$ bombay
~~~

프로젝트 생성 과정에서는 프로젝트 이름, 임베딩 모델, 질의 모델, 벡터 데이터베이스 등을 선택할 수 있습니다. 추후 세부기능을 업데이트 할 예정이며 현재 간단하게 지원되는 템플릿은 다음과 같습니다:

- Basic: 최소한의 설정으로 기본적인 파이프라인을 생성합니다.
- Chatbot: 대화형 단일 채팅 기능을 포함한 파이프라인을 생성합니다.
- Web App: FastAPI를 사용하여 웹 애플리케이션으로 파이프라인을 생성합니다.

생성된 프로젝트는 다음과 같은 디렉토리 구조를 가집니다:

~~~bash
<project_name>/
├── main.py
└── .env
~~~

`main.py` 파일은 선택한 템플릿에 따라 생성된 RAG 파이프라인 코드를 포함하고 있습니다. `.env` 파일에는 환경변수 키가 저장됩니다.

### 2. Bombay 파이프라인 이해하기

`create_pipeline()` 함수를 사용하여 Bombay 파이프라인을 생성합니다. 이 함수는 Bombay 파이프라인의 구성 요소를 설정하고 초기화하는 역할을 합니다. 다음 매개변수를 설정할 수 있습니다:

- `embedding_model_name`: 사용할 임베딩 모델의 이름입니다. 현재는 'openai'만 지원되며, OpenAI의 text-embedding-ada-002 모델을 사용합니다. 이 모델은 텍스트를 고정 길이의 벡터로 변환하는 역할을 합니다.
- `query_model_name`: 사용할 질의 모델의 이름입니다. 현재는 'gpt-3'만 지원되며, OpenAI의 gpt-3.5-turbo 모델을 사용합니다. 이 모델은 검색된 문서를 기반으로 질의에 대한 응답을 생성하는 역할을 합니다.
- `vector_db`: 사용할 벡터 데이터베이스의 이름입니다. 'hnswlib' 또는 'chromadb'를 선택할 수 있습니다. 벡터 데이터베이스는 임베딩 벡터를 저장하고 유사도 기반 검색을 수행하는 역할을 합니다.
- `api_key`: OpenAI API 키를 할당합니다. OpenAI의 임베딩 모델과 질의 모델을 사용하기 위해 필요합니다.
- `similarity`: 유사도 측정 방식을 설정합니다. 기본값은 'cosine'입니다.
- `use_persistent_storage`: ChromaDB 사용 시, 데이터 지속성 여부를 설정합니다. 기본값은 False입니다. True로 설정하면 ChromaDB가 데이터를 디스크에 영구적으로 저장하여 프로그램 종료 후에도 데이터를 유지할 수 있습니다.

예시:

~~~python
from bombay.pipeline import create_pipeline

pipeline = create_pipeline(
    embedding_model_name='openai',
    query_model_name='gpt-3',
    vector_db='chromadb',
    api_key='YOUR_API_KEY',
    similarity='cosine',
    use_persistent_storage=True
)
~~~

위 예시에서는 OpenAI의 임베딩 모델과 GPT-3 질의 모델을 사용하고, ChromaDB를 벡터 데이터베이스로 설정하며, 유사도를 'cosine'으로 설정하고 데이터 지속성을 활성화하여 Bombay 파이프라인을 생성합니다.

### 3. 문서 추가

생성된 Bombay 파이프라인에 문서를 추가하려면 `add_documents()` 메서드를 사용합니다. 이 메서드는 문서 리스트를 입력받아 임베딩을 생성하고 벡터 데이터베이스에 저장합니다.

예시:

~~~python
documents = [
    "고양이는 포유류에 속하는 동물입니다.",
    "고양이는 약 6,000년 전부터 인간과 함께 살아온 것으로 추정됩니다.",
    "고양이는 예민한 청각과 후각을 가지고 있어 작은 움직임이나 냄새도 쉽게 감지할 수 있습니다.",
    "고양이는 앞발에 5개, 뒷발에 4개의 발가락이 있습니다.",
    "고양이는 수면 시간이 많아 하루 평균 15~20시간을 잡니다.",
    "고양이는 점프력이 뛰어나 자신의 몸길이의 최대 6배까지 뛰어오를 수 있습니다."
]

pipeline.add_documents(documents)
~~~

위 예시에서는 고양이에 대한 6개의 문서를 리스트 형태로 정의하고, `add_documents()` 메서드를 사용하여 Bombay 파이프라인에 추가합니다.

### 4. 검색 및 응답 생성

`run_pipeline()` 함수를 사용하여 질의에 대한 응답을 생성합니다. 이 함수는 Bombay 파이프라인의 전체 프로세스를 실행하며, 다음 단계를 수행합니다:

1. 질의 임베딩: 입력된 질의를 임베딩 모델을 사용하여 벡터로 변환합니다.
2. 관련 문서 검색: 질의 벡터를 사용하여 벡터 데이터베이스에서 유사도가 높은 문서를 검색합니다.
3. 응답 생성: 검색된 관련 문서를 기반으로 질의 모델을 사용하여 응답을 생성합니다.

예시:

~~~python
query1 = "고양이는 어떤 동물인가요?"
result1 = run_pipeline(pipeline, documents, query1, k=2)

print("Query 1:")
print(f"Question: {result1['query']}")
print(f"Relevant Documents: {result1['relevant_docs']}")
print(f"Answer: {result1['answer']}")

query2 = "고양이의 수면 시간은 어떻게 되나요?"
result2 = run_pipeline(pipeline, documents, query2, k=1)

print("\nQuery 2:")
print(f"Question: {result2['query']}")
print(f"Relevant Documents: {result2['relevant_docs']}")
print(f"Answer: {result2['answer']}")
~~~

위 예시에서는 두 개의 질의를 정의하고, `run_pipeline()` 함수를 사용하여 각 질의에 대한 응답을 생성합니다.

첫 번째 질의 "고양이는 어떤 동물인가요?"에 대해서는 `k=2`로 설정하여 상위 2개의 관련 문서를 검색하고, 검색된 문서를 기반으로 응답을 생성합니다.

두 번째 질의 "고양이의 수면 시간은 어떻게 되나요?"에 대해서는 `k=1`로 설정하여 가장 유사한 1개의 관련 문서를 검색하고, 검색된 문서를 기반으로 응답을 생성합니다.

실행 결과 예시:

~~~
Query 1:
Question: 고양이는 어떤 동물인가요?
Relevant Documents: ['고양이는 포유류에 속하는 동물입니다.', '고양이는 약 6,000년 전부터 인간과 함께 살아온 것으로 추정됩니다.']
Answer: 고양이는 포유류에 속하는 동물로, 약 6,000년 전부터 인간과 함께 살아온 것으로 추정됩니다.

Query 2:
Question: 고양이의 수면 시간은 어떻게 되나요?
Relevant Documents: ['고양이는 수면 시간이 많아 하루 평균 15~20시간을 잡니다.']
Answer: 고양이는 수면 시간이 많아 하루 평균 15~20시간을 잡니다.
~~~

## 설계 원칙 및 패턴

본 Bombay 파이프라인은 다음과 같은 소프트웨어 공학 원칙과 디자인 패턴을 활용하여 설계되었습니다:

### 추상화(Abstraction)와 인터페이스(Interface)
- `VectorDB`, `EmbeddingModel`, `QueryModel` 추상 클래스를 정의하여 벡터 데이터베이스, 임베딩 모델, 질의 모델에 대한 추상화 제공
- 다양한 구현체를 유연하게 사용할 수 있도록 인터페이스 설계

### 팩토리 패턴(Factory Pattern)
- `create_pipeline` 함수는 팩토리 패턴의 역할 수행
- 주어진 인자에 따라 적절한 임베딩 모델, 질의 모델, 벡터 데이터베이스 선택하여 RAG 파이프라인 생성

### 어댑터 패턴(Adapter Pattern)
- `OpenAIEmbedding`과 `OpenAIQuery` 클래스는 어댑터 패턴 사용
- OpenAI의 API를 추상화된 인터페이스에 맞게 적용

