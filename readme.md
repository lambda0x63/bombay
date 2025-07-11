# Bombay Pipeline

Bombay is a pipeline system for building and utilizing RAG (Retrieval-Augmented Generation) based LLM (Large Language Model).
It guarantees stable operation on Python 3.12 or higher versions.

## Key Features

- **Multiple Model Support**: Supports OpenAI Embedding models and GPT models
- **Vector Database Integration**: Supports Hnswlib and ChromaDB
- **Document Management**: Provides document CRUD functionality through unified interface

## Installation

```bash
pip install bombay
```

## Usage

### Project Creation

```bash
bombay
```

#### Supported Templates

- Basic: Basic pipeline configuration
- Chatbot: Includes interactive single chat functionality
- Web App: FastAPI-based web application

#### Project Structure

```
<project_name>/
├── main.py
└── .env
```

### Pipeline Configuration

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

#### Parameters

- `embedding_model_name`: Embedding model name (currently supports 'openai')
- `query_model_name`: Query model name (currently supports 'gpt-3')
- `vector_db`: Vector database ('hnswlib' or 'chromadb')
- `api_key`: OpenAI API key
- `similarity`: Similarity measurement method (default: 'cosine')
- `use_persistent_storage`: Data persistence option (default: False)

### Adding Documents

```python
documents = [
    "Cats are mammals.",
    "Cats are estimated to have lived with humans for about 6,000 years.",
    "Cats have sensitive hearing and smell, allowing them to easily detect small movements or odors.",
    "Cats have 5 toes on their front paws and 4 toes on their back paws.",
    "Cats sleep a lot, averaging 15-20 hours per day.",
    "Cats have excellent jumping ability and can leap up to 6 times their body length."
]

pipeline.add_documents(documents)
```

### Search and Response Generation

```python
query = "What kind of animal is a cat?"
result = run_pipeline(pipeline, documents, query, k=2)

print(f"Question: {result['query']}")
print(f"Relevant documents: {result['relevant_docs']}")
print(f"Answer: {result['answer']}")
```

#### Execution Result

```
Question: What kind of animal is a cat?
Relevant documents: ['Cats are mammals.', 'Cats are estimated to have lived with humans for about 6,000 years.']
Answer: Cats are mammals that are estimated to have lived with humans for about 6,000 years.
```

## Design Principles

- **Abstraction and Interface**: Abstract class definitions for vector databases, embedding models, and query models
- **Factory Pattern**: Pipeline component creation through `create_pipeline` function
- **Adapter Pattern**: OpenAI API adapted to abstracted interface
