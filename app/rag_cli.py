# CLI app (one-command run)

import sys
from rich import print as rprint
from rich.prompt import Prompt
from rich.console import Console
from .rag_core import init_app_state, answer_question

def main() -> None:
    rprint("[bold cyan]📚 RAG Study Buddy (Gemini + local embeddings)[/bold cyan]")
    rprint("Type your question about the docs. Type [yellow]exit[/yellow] to quit.\n")

    try:
        state = init_app_state()
    except Exception as e:
        rprint(f"[red]Failed to initialize app:[/red] {e}")
        sys.exit(1)

    while True:
        try:
            user_input = Prompt.ask("[bold magenta]You[/bold magenta]")
        except (EOFError, KeyboardInterrupt):
            rprint("\n[green]Goodbye![/green]")
            break

        if not user_input.strip():
            continue
        if user_input.strip().lower() in {"exit", "quit"}:
            rprint("[green]Goodbye![/green]")
            break

        rprint("[dim]Thinking with RAG…[/dim]")
        answer = answer_question(state, user_input)
        console = Console(width=120, soft_wrap=True)
        console.print(f"[bold green]Assistant:[/bold green] {answer}\n")

if __name__ == "__main__":
    main()
