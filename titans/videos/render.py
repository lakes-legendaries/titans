"""CLI for rendering videos

This is run in the docker contain to do the actual rendering
"""

from argparse import ArgumentParser
import os
from pathlib import Path

import sh


# cli
if __name__ == '__main__':

    # parse cli
    parser = ArgumentParser(description='render videos')
    parser.add_argument('--fname', help='File to render')
    parser.add_argument(
        '--first_frame',
        type=int,
        help='First frame to render',
    )
    parser.add_argument(
        '--final_frame',
        type=int,
        help='Last frame to render (inclusive)',
    )
    args = parser.parse_args()

    # check env
    conn_key = 'AZURE_STORAGE_CONNECTION_STRING'
    if conn_key not in os.environ:
        raise OSError(f'env var {conn_key} missing: cannot connect to azure')

    # load in assets
    containers = ['assets', 'blend']
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

    # hard-coded parameters
    odir = 'rendered'

    # run blender
    Path(odir).mkdir(exist_ok=True)
    sh.Command('/blender/blender')([
        '-b',
        f'blend/{args.fname}.blend',
        '--render-output',
        f'{odir}/{args.fname}',
        '-s',
        f'{args.first_frame}',
        '-e',
        f'{args.final_frame}',
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
