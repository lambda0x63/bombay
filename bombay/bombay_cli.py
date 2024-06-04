# bombay/bombay_cli.py
import os
import argparse
from bombay.pipeline import bombay
import pyfiglet
from termcolor import colored

def print_welcome_message():
    ascii_banner = pyfiglet.figlet_format("BOMBAY CLI", font="slant")
    print(colored(ascii_banner, 'cyan'))
    print(colored("Welcome to the Bombay CLI Project Creator!", 'yellow'))
    print(colored("="*50, 'yellow'))
    print(colored("        Let's create something amazing!        ", 'yellow'))
    print(colored("="*50, 'yellow'))
    print()

def select_option(prompt: str, options: list) -> str:
    print(colored(prompt, 'green'))
    for i, option in enumerate(options):
        print(f"{i + 1}. {option}")
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

    project_name = input(colored("Enter project name: ", 'blue'))

    embedding_model = select_option("Select embedding model:", ["OpenAI"])
    query_model = select_option("Select query model:", ["GPT-3.5"])
    vector_db = select_option("Select vector database:", ["ChromaDB", "Hnswlib"])

    api_key = input(colored("Enter OpenAI API key (leave blank to set later): ", 'blue'))
    if not api_key:
        api_key = "your-api-key-here"

    print(colored(f"\nProject name: {project_name}", 'magenta'))
    print(colored("Creating project...", 'magenta'))

    os.makedirs(project_name, exist_ok=True)

    main_py_content = f"""from bombay.pipeline.bombay import create_pipeline

# basic pipeline
pipeline = create_pipeline(
    embedding_model_name='{embedding_model}',
    query_model_name='{query_model}',
    vector_db='{vector_db}',
    api_key='{api_key}'
)

"""

    with open(f"{project_name}/main.py", "w", encoding="utf-8") as f:
        f.write(main_py_content)

    print(colored("\nProject created successfully!", 'green'))
    print(colored("="*50, 'yellow'))
    print(colored("                 Next steps                 ", 'yellow'))
    print(colored("="*50, 'yellow'))
    print(colored(f"1. cd {project_name}", 'cyan'))
    print(colored("2. Modify main.py to add your documents and perform searches", 'cyan'))
    print(colored("3. Run 'python main.py' to execute the project", 'cyan'))

def main():
    parser = argparse.ArgumentParser(description="Bombay CLI tool")
    subparsers = parser.add_subparsers(dest='command')

    # create subcommand
    create_parser = subparsers.add_parser('create', help='Create a new Bombay project')
    create_parser.set_defaults(func=create_project)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
    else:
        args.func()

if __name__ == "__main__":
    main()
