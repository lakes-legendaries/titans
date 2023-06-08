"""CLI for running blender in the built docker container"""

import os
from os.path import join
from pathlib import Path
import re

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


@app.command()
def convert(fname: str = typer.Option(...)):
    """Convert mkv to modern video container formats

    Parameters
    ----------
    fname: str
        input video filename
    """

    # download files
    _download_containers("rendered")

    # get output filename
    odir = "videos"
    ifname = join("rendered", fname) + ".mkv"
    ofname = join(
        odir,
        (
            re.sub(
                r"[^a-zA-Z0-9.]",
                r"_",
                fname,
            ).lower()
            .removesuffix(".mkv")
        ),
    )
    Path(odir).mkdir(exist_ok=True)

    # convert to H.264
    common_ffmpeg = [
        "-i",
        ifname,
        "-strict",
        "-2",
        "-y",
        "-c:v",
    ]
    sh.ffmpeg(
        *common_ffmpeg,
        "libx264",
        "-c:a",
        "aac",
        ofname + ".h264.mp4",
    )

    # convert to H.265
    sh.ffmpeg(
        *common_ffmpeg,
        "libx265",
        "-vtag",
        "hvc1",
        ofname + ".h265.mp4",
    )

    # convert to avi
    common_docker = [
        "--rm",
        "-v",
        f"{os.getcwd()}:{os.getcwd()}",
        "-w",
        os.getcwd(),
    ]
    sh.docker.run(
        *common_docker,
        "mwader/static-ffmpeg:5.0.1-3",
        *common_ffmpeg,
        "libsvtav1",
        "-preset",
        "4",
        ofname + ".av1.mp4",
    )
    sh.docker.run(
        *common_docker,
        "debian:stable-slim",
        "/bin/bash",
        "-c",
        f'"chmod 777 {ofname}.av1.mp4"',
    )

    # convert to webm
    common_webm = [
        *common_ffmpeg,
        "libvpx-vp9",
        "-b:v",
        "0",
        "-crf",
        "50",
        "-pass",
    ]
    sh.ffmpeg(
        *common_webm,
        "1",
        "-an",
        "-f",
        "null",
        "/dev/null",
    )
    sh.ffmpeg(
        *common_webm,
        "2",
        "-c:a",
        "libopus",
        ofname + ".vp9.webm",
    )

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
