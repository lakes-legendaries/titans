"""CLI for running blender in the built docker container"""

import os
from os.path import join
from pathlib import Path

import sh
import typer


# create cli
app = typer.Typer()


def _download_containers(*containers: str):
    """Download containers from azure storage

    Parameters
    ----------
    *containers : str
        containers to download
    """
    for container in containers:
        Path(container).mkdir(exist_ok=True)
        sh.azcopy.copy([
            (
                "https://titansfileserver.blob.core.windows.net/"
                + f"{container}/*{os.environ['AZCOPY_SAS']}"
            ),
            f"{container}",
            "--recursive",
        ])


@app.command()
def animate(
    fname: str = typer.Option(...),
    first_frame: int = typer.Option(...),
    final_frame: int = typer.Option(...),
):
    """Animate blender file

    Parameters
    ----------
    fname : str
        Blender file to animate
    first_frame : int
        First frame to animate
    final_frame : int
        Last frame to animate (inclusive)
    """

    # download assets and blender files
    _download_containers("assets", "blend")

    # run blender
    odir = "animated"
    ofname = f'{odir}/{fname}'
    Path(odir).mkdir(exist_ok=True)
    sh.Command('/blender/blender')([
        '-b',
        f'blend/{fname}.blend',
        '--render-output',
        ofname,
        '-s',
        f'{first_frame}',
        '-e',
        f'{final_frame}',
        '-a',
    ])

    # upload result
    sh.azcopy.copy([
        f"{ofname}*",
        (
            "https://titansfileserver.blob.core.windows.net/"
            + f"{odir}/{os.environ['AZCOPY_SAS']}"
        ),
        "--recursive",
    ])


@app.command()
def render(
    fname: str = typer.Option(...),
    containers: list[str] = typer.Option(...),
    mkv: bool = typer.Option(...),
):
    """Render blender videos

    Parameters
    ----------
    fname: str
        Blender file to render
    containers: list[str]
        containers to download for task
    mkv: bool
        whether to render as mkv (instead of png)
    """

    # download files
    _download_containers(*containers)

    # run blender
    odir = "rendered"
    ofname = join(odir, fname) + (".mkv" if mkv else "")
    Path(odir).mkdir(exist_ok=True)
    sh.Command('/blender/blender')([
        '-b',
        f'blend/{fname}.blend',
        '--render-output',
        ofname,
        '-a',
    ])

    # upload result
    sh.azcopy.copy([
        f"{ofname}*",
        (
            "https://titansfileserver.blob.core.windows.net/"
            + f"{odir}/{os.environ['AZCOPY_SAS']}"
        ),
        "--recursive",
    ])


# cli
if __name__ == '__main__':

    # check env
    conn_key = "AZCOPY_SAS"
    if conn_key not in os.environ:
        raise OSError(f'env var {conn_key} missing: cannot connect to azure')

    # run app
    app()
