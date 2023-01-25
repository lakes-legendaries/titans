"""CLI for rendering videos"""

from argparse import ArgumentParser
import os
from pathlib import Path
from subprocess import run


# cli
if __name__ == '__main__':

    # parse cli
    parser = ArgumentParser(description='render Titans Of Eden videos')
    args = parser.parse_args()

    # check env
    conn_key = 'AZURE_STORAGE_CONNECTION_STRING'
    if conn_key not in os.environ:
        raise OSError(f'env var {conn_key} missing: cannot connect to azure')

    # load in assets
    containers = ['assets', 'blend']
    for container in containers:
        Path(container).mkdir(exist_ok=True)
        run(
            [
                'az',
                'storage',
                'blob',
                'download-batch',
                '--source',
                container,
                '--destination',
                container,
                '--overwrite',
            ],
            capture_output=True,
            check=True,
        )

    # run blender
    odir = 'rendered'
    Path(odir).mkdir(exist_ok=True)
    run(
        [
            '/blender/blender',
            '-b',
            'blend/Storm Title.blend',
            '--render-output',
            f'{odir}/Storm Title',
            '-s',
            '0',
            '-e',
            '0',
            '-a',
        ],
        capture_output=True,
        check=True,
    )

    # upload result
    run(
        [
            'az',
            'storage',
            'blob',
            'upload-batch',
            '-s',
            odir,
            '-d',
            odir,
            '--overwrite',
        ],
        # capture_output=True,
        check=True,
    )
