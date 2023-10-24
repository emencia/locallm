import click

from locallm import __version__


@click.command()
@click.pass_context
def version_command(context):
    """
    Print out version information.
    """
    click.echo("locallm {}".format(__version__))
