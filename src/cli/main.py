"""Main CLI interface for the journal query system."""

import click
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple
import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress
from rich.markdown import Markdown

from src.application.engine import QueryEngine
from src.application.config import Config

console = Console()


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--data-path', '-d', type=click.Path(), help='Path to journal files')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def cli(ctx, config, data_path, verbose):
    """Personal Journal Query System - Query your journal entries using AI."""
    ctx.ensure_object(dict)

    # Load configuration
    if config:
        config_obj = Config.load_from_file(Path(config))
    else:
        config_obj = Config()

    if data_path:
        config_obj.data_path = Path(data_path)

    if verbose:
        config_obj.log_level = "DEBUG"

    ctx.obj['config'] = config_obj
    ctx.obj['engine'] = QueryEngine(config_obj)


@cli.command()
@click.option('--force', '-f', is_flag=True, help='Force rebuild of indexes')
@click.pass_context
def index(ctx, force):
    """Index journal entries for searching."""
    engine = ctx.obj['engine']

    # Show model info on first run
    config = ctx.obj['config']
    if config.llm.model_type == "mock":
        console.print("\n[yellow]ℹ️  Using mock AI model for immediate functionality.[/yellow]")
        console.print("For better responses, download a local model:")
        console.print("  [cyan]python main.py download-model[/cyan]")
        console.print("  [cyan]export JOURNAL_LLM_MODEL_TYPE=local[/cyan]\n")

    with Progress() as progress:
        task = progress.add_task("[green]Indexing journal entries...", total=None)

        try:
            stats = engine.ingest_data(force_rebuild=force)

            progress.update(task, completed=True)

            # Display results
            table = Table(title="Indexing Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("Total Entries", str(stats['total_entries']))
            if 'date_range' in stats and stats['date_range']['earliest']:
                table.add_row("Date Range", f"{stats['date_range']['earliest']} to {stats['date_range']['latest']}")
            if 'entry_types' in stats:
                types_str = ", ".join(f"{k}: {v}" for k, v in stats['entry_types'].items())
                table.add_row("Entry Types", types_str)

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error during indexing: {e}[/red]")


@cli.command()
@click.argument('query')
@click.option('--limit', '-l', default=10, help='Number of results to return')
@click.option('--since', help='Start date (YYYY-MM-DD)')
@click.option('--until', help='End date (YYYY-MM-DD)')
@click.option('--type', 'response_type', default='question_answering',
              type=click.Choice(['question_answering', 'summarization', 'reflection', 'emotion_analysis']))
@click.option('--sources', '-s', is_flag=True, help='Show sources')
@click.option('--format', 'output_format', default='rich',
              type=click.Choice(['rich', 'plain', 'json']))
@click.pass_context
def query(ctx, query, limit, since, until, response_type, sources, output_format):
    """Query journal entries."""
    engine = ctx.obj['engine']

    # Parse date filters
    date_filter = None
    if since or until:
        start_date = datetime.fromisoformat(since) if since else datetime.min
        end_date = datetime.fromisoformat(until) if until else datetime.max
        date_filter = (start_date, end_date)

    try:
        result = engine.query(
            query,
            k=limit,
            date_filter=date_filter,
            response_type=response_type
        )

        if output_format == 'json':
            output = {
                'query': result.query,
                'response': result.response,
                'sources': [
                    {
                        'id': s.entry.id,
                        'title': s.entry.title,
                        'date': s.entry.date.isoformat(),
                        'score': s.score
                    }
                    for s in result.sources
                ],
                'processing_time': result.processing_time
            }
            console.print(json.dumps(output, indent=2))

        elif output_format == 'plain':
            console.print(f"Query: {result.query}")
            console.print(f"Response: {result.response}")
            if sources and result.sources:
                console.print("\nSources:")
                for i, source in enumerate(result.sources, 1):
                    date_str = source.entry.date.strftime("%Y-%m-%d")
                    console.print(f"{i}. [{date_str}] {source.entry.title}")

        else:  # rich format
            # Display response
            response_panel = Panel(
                Markdown(result.response),
                title=f"Response to: {result.query}",
                border_style="green"
            )
            console.print(response_panel)

            # Display sources if requested
            if sources and result.sources:
                table = Table(title="Sources")
                table.add_column("Date", style="cyan")
                table.add_column("Title", style="magenta")
                table.add_column("Score", style="yellow")

                for source in result.sources:
                    date_str = source.entry.date.strftime("%Y-%m-%d")
                    title = source.entry.title or "Untitled"
                    score = f"{source.score:.3f}"
                    table.add_row(date_str, title, score)

                console.print(table)

            # Display metadata
            console.print(f"\n[dim]Processing time: {result.processing_time:.2f}s | "
                         f"Model: {result.model_used} | "
                         f"Sources: {len(result.sources)}[/dim]")

    except Exception as e:
        console.print(f"[red]Error during query: {e}[/red]")


@cli.command()
@click.option('--days', '-d', default=30, help='Number of days to analyze')
@click.option('--since', help='Start date (YYYY-MM-DD)')
@click.option('--until', help='End date (YYYY-MM-DD)')
@click.pass_context
def emotions(ctx, days, since, until):
    """Analyze emotions in journal entries."""
    engine = ctx.obj['engine']

    try:
        if since or until:
            start_date = datetime.fromisoformat(since) if since else datetime.min
            end_date = datetime.fromisoformat(until) if until else datetime.max
        else:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

        result = engine.analyze_emotions(start_date, end_date)

        panel = Panel(
            Markdown(result.response),
            title=f"Emotional Analysis",
            border_style="blue"
        )
        console.print(panel)

    except Exception as e:
        console.print(f"[red]Error during emotion analysis: {e}[/red]")


@cli.command()
@click.option('--days', '-d', default=7, help='Number of days to summarize')
@click.option('--since', help='Start date (YYYY-MM-DD)')
@click.option('--until', help='End date (YYYY-MM-DD)')
@click.option('--focus', help='Focus topic for summary')
@click.pass_context
def summary(ctx, days, since, until, focus):
    """Summarize journal entries for a time period."""
    engine = ctx.obj['engine']

    try:
        if since or until:
            start_date = datetime.fromisoformat(since) if since else datetime.min
            end_date = datetime.fromisoformat(until) if until else datetime.max
        else:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

        result = engine.summarize_period(start_date, end_date, focus)

        period_str = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        title = f"Summary: {period_str}"
        if focus:
            title += f" (Focus: {focus})"

        panel = Panel(
            Markdown(result.response),
            title=title,
            border_style="cyan"
        )
        console.print(panel)

    except Exception as e:
        console.print(f"[red]Error during summarization: {e}[/red]")


@cli.command()
@click.argument('goal')
@click.option('--days', '-d', default=90, help='Number of days to look back')
@click.pass_context
def goal(ctx, goal, days):
    """Track progress on a specific goal."""
    engine = ctx.obj['engine']

    try:
        result = engine.track_goal(goal, days_back=days)

        panel = Panel(
            Markdown(result.response),
            title=f"Goal Progress: {goal}",
            border_style="yellow"
        )
        console.print(panel)

    except Exception as e:
        console.print(f"[red]Error during goal tracking: {e}[/red]")


@cli.command()
@click.pass_context
def stats(ctx):
    """Show system statistics."""
    engine = ctx.obj['engine']

    try:
        stats = engine.get_statistics()

        table = Table(title="System Statistics")
        table.add_column("Component", style="cyan")
        table.add_column("Metric", style="magenta")
        table.add_column("Value", style="yellow")

        # Metadata stats
        meta_stats = stats['metadata']
        table.add_row("Database", "Total Entries", str(meta_stats['total_entries']))
        table.add_row("Database", "Unique Files", str(meta_stats['unique_files']))
        table.add_row("Database", "Query Count", str(meta_stats['query_count']))

        # Vector store stats
        vector_stats = stats['vector_store']
        table.add_row("Vector Store", "Total Vectors", str(vector_stats['total_vectors']))
        table.add_row("Vector Store", "Dimension", str(vector_stats['dimension']))
        table.add_row("Vector Store", "Memory (MB)", f"{vector_stats['memory_usage_mb']:.1f}")

        # Model info
        embed_info = stats['embedding_model']
        table.add_row("Embedding", "Model", embed_info['model_name'])
        table.add_row("Embedding", "Dimension", str(embed_info['dimension']))

        llm_info = stats['llm_model']
        table.add_row("LLM", "Model", llm_info['model_name'])
        table.add_row("LLM", "Backend", llm_info.get('backend', 'unknown'))

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error getting statistics: {e}[/red]")


@cli.command()
@click.option('--model', default='phi-2', help='Model to download (phi-2, llama2-7b)')
@click.pass_context
def download_model(ctx, model):
    """Download a local language model for better responses."""
    from src.generation.llm import LocalLLM
    from pathlib import Path

    try:
        console.print(f"[yellow]Downloading {model} model...[/yellow]")
        console.print("This may take a few minutes depending on your internet connection.")

        if model == 'phi-2':
            # Create a LocalLLM instance which will trigger download
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)

            model_path = models_dir / "phi-2.Q4_K_M.gguf"
            if model_path.exists():
                console.print(f"[green]Model {model} already exists at {model_path}[/green]")
                return

            # This will trigger the download
            try:
                llm = LocalLLM()
                console.print(f"[green]Successfully downloaded {model} model![/green]")
                console.print("You can now use local models by setting:")
                console.print("  export JOURNAL_LLM_MODEL_TYPE=local")
                console.print("  or updating your config file")
            except Exception as e:
                console.print(f"[red]Download failed: {e}[/red]")
                console.print("You can continue using the mock model or try Ollama:")
                console.print("  brew install ollama")
                console.print("  ollama pull llama2")
                console.print("  export JOURNAL_LLM_MODEL_TYPE=ollama")

        else:
            console.print(f"[red]Unknown model: {model}[/red]")
            console.print("Available models: phi-2")

    except Exception as e:
        console.print(f"[red]Error downloading model: {e}[/red]")


@cli.command()
@click.pass_context
def config_show(ctx):
    """Show current configuration."""
    config = ctx.obj['config']

    config_dict = config.model_dump()
    config_json = json.dumps(config_dict, indent=2, default=str)

    panel = Panel(
        config_json,
        title="Current Configuration",
        border_style="green"
    )
    console.print(panel)


@cli.command()
@click.option('--output', '-o', required=True, help='Output file path')
@click.pass_context
def config_save(ctx, output):
    """Save current configuration to file."""
    config = ctx.obj['config']
    output_path = Path(output)

    try:
        config.save_to_file(output_path)
        console.print(f"[green]Configuration saved to {output_path}[/green]")
    except Exception as e:
        console.print(f"[red]Error saving configuration: {e}[/red]")


def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()