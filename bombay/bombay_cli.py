# bombay/bombay_cli.py
import os
import argparse
import pyfiglet
from termcolor import colored
from bombay.pipeline import create_pipeline

def print_welcome_message():
    ascii_banner = pyfiglet.figlet_format("BOMBAY CLI", font="slant")
    print(colored(ascii_banner, 'cyan'))
    print(colored("Welcome to the Bombay CLI Project Creator!", 'yellow'))
    print(colored("="*50, 'yellow'))
    print(colored("        Let's create RAG System simply!        ", 'yellow'))
    print(colored("="*50, 'yellow'))
    print()

def select_option(prompt: str, options: list) -> str:
    print(colored(prompt, 'green'))
    for i, option in enumerate(options):
        print(colored(f"{i + 1}. {option}", 'white'))
    while True:
        try:
            choice = int(input(colored("Select an option: ", 'blue'))) - 1
            if 0 <= choice < len(options):
                return options[choice]
            else:
                print(colored("Invalid option. Please select a valid number.", 'red'))
        except ValueError:
            print(colored("Invalid input. Please enter a number.", 'red'))

def create_project():
    """Create a new Bombay project."""
    print_welcome_message()

    project_name = input(colored("Enter project name: ", 'blue')).strip()

    embedding_model = select_option("Select embedding model:", ["openai"])
    query_model = select_option("Select query model:", ["gpt-3"])
    vector_db = select_option("Select vector database:", ["chromadb", "hnswlib"])

    api_key = input(colored("Enter OpenAI API key (leave blank to set later): ", 'blue')).strip()
    if not api_key:
        api_key = "your-api-key-here"

    print(colored(f"\nProject name: {project_name}", 'magenta'))
    print(colored("Creating project...", 'magenta'))

    os.makedirs(project_name, exist_ok=True)

    main_py_content = f"""from bombay.pipeline import create_pipeline

# Basic pipeline
pipeline = create_pipeline(
    embedding_model_name='{embedding_model}',
    query_model_name='{query_model}',
    vector_db='{vector_db}',
    api_key='{api_key}'
)

"""

    with open(f"{project_name}/main.py", "w", encoding="utf-8") as f:
        f.write(main_py_content)

    # Create example file
    create_example_file(project_name, embedding_model, query_model, vector_db, api_key)

    print(colored("\nProject created successfully!", 'green'))
    print(colored("="*50, 'yellow'))
    print(colored("                 Next steps                 ", 'yellow'))
    print(colored("="*50, 'yellow'))
    print(colored(f"1. cd {project_name}", 'cyan'))
    print(colored("2. Run 'pip install bombay' to install the Bombay package", 'cyan'))
    print(colored("3. Modify main.py to add your documents and perform searches", 'cyan'))
    print(colored("4. Run 'python main.py' to execute the project", 'cyan'))

def create_example_file(project_name, embedding_model, query_model, vector_db, api_key):
    example_content = f"""from bombay.pipeline import create_pipeline, run_pipeline

# OpenAI API key setup
api_key = '{api_key}'

# Example 1: Using ChromaDB
pipeline_chromadb = create_pipeline(
    embedding_model_name='{embedding_model}',
    query_model_name='{query_model}',
    vector_db='chromadb',
    api_key=api_key,
    similarity='cosine',
    use_persistent_storage=False
)

# Example 2: Using Hnswlib
pipeline_hnswlib = create_pipeline(
    embedding_model_name='{embedding_model}',
    query_model_name='{query_model}',
    vector_db='hnswlib',
    api_key=api_key,
    similarity='cosine'
)

# Add documents
documents = [
    "Cats are mammals.",
    "Cats have lived with humans for about 6,000 years.",
    "Cats have sensitive hearing and smell."
]

pipeline_chromadb.add_documents(documents)
pipeline_hnswlib.add_documents(documents)

# Query
query = "What kind of animal is a cat?"
result_chromadb = run_pipeline(pipeline_chromadb, documents, query, k=1)
result_hnswlib = run_pipeline(pipeline_hnswlib, documents, query, k=1)

# Output results for ChromaDB
print("Results using ChromaDB:")
print(f"Query: {{result_chromadb['query']}}")
print(f"Relevant Documents: {{result_chromadb['relevant_docs']}}")
print(f"Distances: {{result_chromadb['distances']}}")
print(f"Answer: {{result_chromadb['answer']}}")

# Output results for Hnswlib
print("\\nResults using Hnswlib:")
print(f"Query: {{result_hnswlib['query']}}")
print(f"Relevant Documents: {{result_hnswlib['relevant_docs']}}")
print(f"Distances: {{result_hnswlib['distances']}}")
print(f"Answer: {{result_hnswlib['answer']}}")
"""

    with open(f"{project_name}/example.py", "w", encoding="utf-8") as f:
        f.write(example_content)

def main():
    parser = argparse.ArgumentParser(description="Bombay CLI tool")
    subparsers = parser.add_subparsers(dest='command')

    create_parser = subparsers.add_parser('create', help='Create a new Bombay project')
    create_parser.set_defaults(func=create_project)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
    else:
        args.func()

if __name__ == "__main__":
    main()
