import logging

import click

from locallm.hello import HelloBase, HelloHTML


@click.command()
@click.option(
    "-f",
    "--format",
    "output_format",
    type=click.Choice(
        ["plain", "html"],
        case_sensitive=False
    ),
    help=(
        "Select output format, default to 'plain'."
    ),
    default="plain",
)
@click.option(
    "-c",
    "--container",
    help=(
        "Define the HTML container name to use around text. "
        "Default to 'p' to make a paragraph."
    ),
    default=None,
)
@click.argument("name", required=False)
@click.pass_context
def greet_command(context, output_format, container, name):
    """
    Greet someone or something.
    """
    logger = logging.getLogger("locallm")

    logger.debug("Required format: {}".format(output_format))
    logger.debug("Required container: {}".format(container))

    if output_format == "plain" and container:
        logger.warning("Defining a HTML container in plain format has no sense.")

    if name == "ass":
        logger.critical("Please do not be so crude.")
        raise click.Abort()

    if output_format == "html":
        builder_cls = HelloHTML
    else:
        builder_cls = HelloBase

    builder = builder_cls(
        name=name,
        container=container,
    )

    click.echo(builder.greet())
