"""CLI for rendering videos"""

from argparse import ArgumentParser
import os
from pathlib import Path
from subprocess import run


# cli
if __name__ == '__main__':

    # parse cli
    parser = ArgumentParser(description='render videos')
    parser.add_argument('--fname', help='File to render')
    parser.add_argument('--frame', type=int, help='Frame to render')
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

    # hard-coded parameters
    odir = 'rendered'

    # run blender
    Path(odir).mkdir(exist_ok=True)
    run(
        [
            '/blender/blender',
            '-b',
            f'blend/{args.fname}.blend',
            '--render-output',
            f'{odir}/{args.fname}',
            '-s',
            f'{args.frame}',
            '-e',
            f'{args.frame}',
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
        capture_output=True,
        check=True,
    )
