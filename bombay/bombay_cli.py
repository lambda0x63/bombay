# bombay/bombay_cli.py
import os
import argparse
import pyfiglet
from rich.console import Console
from rich.prompt import Prompt
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from string import Template
from bombay.templates import get_project_templates

console = Console()

def print_welcome_message():
    ascii_banner = pyfiglet.figlet_format("BOMBAY CLI", font="slant")
    console.print(Text(ascii_banner, justify="center", style="cyan"))
    welcome_panel = Panel(
        "Welcome to the Bombay CLI Project Creator!\nLet's create a RAG System simply!",
        title="Welcome",
        subtitle="Enjoy the process!",
        subtitle_align="center",
        border_style="yellow",
        style="yellow"
    )
    console.print(welcome_panel)

def print_initial_message():
    console.clear()
    print_welcome_message()
    
    choice = select_option("Select an action:", ["Create a new project", "Exit"])
    if choice == "Create a new project":
        create_project()
    else:
        console.print("[yellow]Exiting...[/yellow]")

def select_option(prompt: str, options: list, descriptions: list = None) -> str:
    while True:
        console.clear()
        print_welcome_message()
        console.print(Panel(prompt, style="green", border_style="bold green"))
        options_table = Table(show_header=False)
        options_table.add_column("Option", style="white")
        options_table.add_column("Description", style="cyan")
        
        for i, option in enumerate(options):
            description = descriptions[i] if descriptions else ""
            options_table.add_row(f"{i + 1}. {option}", description)
        
        console.print(options_table)
        try:
            choice = int(Prompt.ask("[blue]Select an option[/blue]")) - 1
            if 0 <= choice < len(options):
                return options[choice]
            else:
                console.print("[red]Invalid option. Please select a valid number.[/red]")
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/red]")

def create_project():
    """Create a new Bombay project."""
    console.clear()
    print_welcome_message()
    project_name = Prompt.ask("[blue]Enter project name[/blue]").strip()

    template_options = {
        "Basic": "A basic pipeline template with minimal configuration.",
        "Chatbot": "A chatbot pipeline template for interactive conversations.",
        "Web App": "A web application pipeline template using FastAPI."
    }

    template = select_option("Select a project template:", list(template_options.keys()), list(template_options.values()))

    embedding_model = select_option("Select embedding model:", ["openai"])
    query_model = select_option("Select query model:", ["gpt-3"])
    vector_db = select_option("Select vector database:", ["chromadb", "hnswlib"])

    if vector_db == "chromadb":
        storage_mode = select_option("Select storage mode:", ["In-Memory", "Persistent"])
        use_persistent_storage = storage_mode == "Persistent"
    else:
        use_persistent_storage = False

    console.clear()
    print_welcome_message()
    
    api_key = Prompt.ask("[blue]Enter OpenAI API key (leave blank to set later)[/blue]", default="your-api-key").strip()

    console.clear()
    print_welcome_message()
    summary_table = Table(title="Project Summary", style="magenta", border_style="magenta")
    summary_table.add_column("Field", style="yellow", justify="right")
    summary_table.add_column("Value", style="cyan", justify="left")
    summary_table.add_row("Project Name", project_name)
    summary_table.add_row("Project Template", template)
    summary_table.add_row("Embedding Model", embedding_model)
    summary_table.add_row("Query Model", query_model)
    summary_table.add_row("Vector Database", vector_db)
    summary_table.add_row("Storage Mode", storage_mode if vector_db == "chromadb" else "In-Memory")
    summary_table.add_row("API Key", api_key)
    
    console.print(summary_table)

    if Prompt.ask("[magenta]Do you want to create the project with these settings?[/magenta]", choices=["y", "n"]) == "y":
        console.print("[magenta]Creating project...[/magenta]")

        os.makedirs(project_name, exist_ok=True)

        template_content = Template(get_project_templates()[template])
        main_py_content = template_content.substitute(
            embedding_model=embedding_model,
            query_model=query_model,
            vector_db=vector_db,
            use_persistent_storage=use_persistent_storage
        )

        with open(f"{project_name}/main.py", "w", encoding="utf-8") as f:
            f.write(main_py_content)

        with open(f"{project_name}/.env", "w") as f:
            f.write(f"OPENAI_API_KEY={api_key}\n")

        console.clear()
        print_welcome_message()
        console.print("\n[green]Project created successfully![/green]")
        next_steps_panel = Panel.fit(
            f"\n1. cd {project_name}\n2. Implement your RAG system in main.py\n3. Run 'python main.py'",
            title="Next steps",
            border_style="yellow",
            style="cyan"
        )
        console.print(next_steps_panel)
    else:
        console.print("[yellow]Project creation canceled.[/yellow]")

def main():
    parser = argparse.ArgumentParser(description="Bombay CLI tool")
    subparsers = parser.add_subparsers(dest='command')

    create_parser = subparsers.add_parser('create', help='Create a new Bombay project')
    create_parser.set_defaults(func=create_project)

    args = parser.parse_args()

    if args.command is None:
        print_initial_message()
    else:
        args.func()

if __name__ == "__main__":
    main()