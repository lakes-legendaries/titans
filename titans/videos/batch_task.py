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
    ifname = join("rendered", fname) + ".mkv"
    ofname = (
        re.sub(
            r"[^a-zA-Z0-9.]",
            r"_",
            fname,
        ).lower()
        .removesuffix(".mkv")
    )

    # function to upload
    def upload(ofname_full: str):
        sh.azcopy.copy([
            f"{ofname_full}",
            (
                "https://titansfileserverdev.blob.core.windows.net/"
                + f"$web/vid/{os.environ['AZCOPY_DEV_SAS']}"
            ),
            "--recursive",
        ])

    # convert to H.264
    ofname_h264 = f"{ofname}.h264.mp4"
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
        ofname_h264,
    )
    upload(ofname_h264)

    # convert to H.265
    ofname_h265 = f"{ofname}.h265.mp4"
    sh.ffmpeg(
        *common_ffmpeg,
        "libx265",
        "-vtag",
        "hvc1",
        ofname_h265,
    )
    upload(ofname_h265)

    # convert to webm
    ofname_webm = f"{ofname}.vp9.webm"
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
        ofname_webm,
    )
    upload(ofname_webm)


# cli
if __name__ == '__main__':

    # check env
    conn_key = "AZCOPY_SAS"
    if conn_key not in os.environ:
        raise OSError(f'env var {conn_key} missing: cannot connect to azure')

    # run app
    app()
