# bombay/templates.py
def get_project_templates():
    return {
        "Basic": """from bombay.pipeline.bombay import create_pipeline, run_pipeline
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# Create a pipeline
pipeline = create_pipeline(
    embedding_model_name='$embedding_model',
    query_model_name='$query_model',
    vector_db='$vector_db',
    api_key=api_key,
    similarity='cosine',
    use_persistent_storage=$use_persistent_storage
)

# Add documents
documents = [
    "Document 1 text goes here...",
    "Document 2 text goes here...",
    "Document 3 text goes here..."
]
pipeline.add_documents(documents)

# Query the pipeline
query = "Your question goes here..."
result = run_pipeline(pipeline, documents, query, k=1)

# Print the results
print(f"Query: {query}")
print(f"Answer: {result['answer']}")
""",

        "Chatbot": """from bombay.pipeline.bombay import create_pipeline, run_pipeline
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# Create a pipeline
pipeline = create_pipeline(
    embedding_model_name='$embedding_model',
    query_model_name='$query_model',
    vector_db='$vector_db',
    api_key=api_key,
    similarity='cosine',
    use_persistent_storage=$use_persistent_storage
)

# Add documents
documents = [
    "Document 1 text goes here...",
    "Document 2 text goes here...",
    "Document 3 text goes here..."
]
pipeline.add_documents(documents)

# Chatbot loop
while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    result = run_pipeline(pipeline, documents, user_input, k=1)
    print(f"Assistant: {result['answer']}")
""",
        "Web App": """from bombay.pipeline.bombay import create_pipeline, run_pipeline
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# Create a pipeline
pipeline = create_pipeline(
    embedding_model_name='$embedding_model',
    query_model_name='$query_model',
    vector_db='$vector_db',
    api_key=api_key,
    similarity='cosine',
    use_persistent_storage=$use_persistent_storage
)

# Add documents
documents = [
    "Document 1 text goes here...",
    "Document 2 text goes here...",
    "Document 3 text goes here..."
]
pipeline.add_documents(documents)

# FastAPI app
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    query = request.query
    try:
        result = run_pipeline(pipeline, documents, query, k=1)
        return {"query": query, "answer": result['answer']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
"""
    }