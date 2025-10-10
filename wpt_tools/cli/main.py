"""CLI entry point for wpt-tools."""

import click


@click.group()
@click.version_option()
def cli():
    """Wireless Power Tools - Analyze wireless power systems."""
    pass


@cli.command()
def demo():
    """Run the wireless power tools demo with sample data."""
    import subprocess
    import sys
    from pathlib import Path

    # Get the path to the example script
    examples_dir = Path(__file__).parent.parent.parent / "examples"
    script_path = examples_dir / "wpt_example.py"

    if not script_path.exists():
        click.echo(f"Demo script not found at {script_path}", err=True)
        sys.exit(1)

    click.echo(f"Running: {script_path}")

    # Run the example script
    try:
        result = subprocess.run([sys.executable, str(script_path)], cwd=examples_dir)
        if result.returncode == 0:
            click.echo("Demo completed successfully!")
        else:
            click.echo(f"Demo failed with exit code {result.returncode}", err=True)
            sys.exit(result.returncode)
    except Exception as e:
        click.echo(f"Error running demo: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
