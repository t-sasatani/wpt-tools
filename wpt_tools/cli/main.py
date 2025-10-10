"""CLI entry point for wpt-tools."""

import click

from wpt_tools.workflow import demo_workflow


@click.group()
@click.version_option()
def cli():
    """Wireless Power Tools - Analyze wireless power systems."""
    pass


@cli.command()
@click.option(
    "--show-plots", is_flag=True, default=False, help="Show interactive plots"
)
def demo(show_plots: bool):
    """Run the wireless power tools demo with sample data."""
    click.echo("Running wireless power tools demo...")
    demo_workflow(show_plot=show_plots)
    click.echo("Demo completed successfully!")


if __name__ == "__main__":
    cli()
