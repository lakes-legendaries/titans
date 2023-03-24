"""CLI for running blender in the built docker container"""

import os
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
        sh.az.storage.blob([
            'download-batch',
            '--source',
            container,
            '--destination',
            container,
            '--overwrite',
        ])


@app.command()
def animate(fname: str, first_frame: int, final_frame: int):
    """Animate blender file

    Parameters
    ----------
    fname : str
        Blender file to render
    first_frame : int
        First frame to render
    final_frame : int
        Last frame to render (inclusive)
    """

    # download assets and blender files
    _download_containers("assets", "blend")

    # run blender
    odir = "rendered"
    Path(odir).mkdir(exist_ok=True)
    sh.Command('/blender/blender')([
        '-b',
        f'blend/{fname}.blend',
        '--render-output',
        f'{odir}/{fname}',
        '-s',
        f'{first_frame}',
        '-e',
        f'{final_frame}',
        '-a',
    ])

    # upload result
    sh.az.storage.blob([
        'upload-batch',
        '-s',
        odir,
        '-d',
        odir,
        '--overwrite',
    ])


# cli
if __name__ == '__main__':

    # check env
    conn_key = 'AZURE_STORAGE_CONNECTION_STRING'
    if conn_key not in os.environ:
        raise OSError(f'env var {conn_key} missing: cannot connect to azure')

    # run app
    app()