"""CLI for rendering videos"""

from argparse import ArgumentParser
import os
from os import remove
from os.path import join
from pathlib import Path
from shutil import rmtree
from subprocess import run


# cli
if __name__ == '__main__':

    # parse cli
    parser = ArgumentParser(description='render Titans Of Eden videos')
    parser.add_argument('--fname', required=True)
    args = parser.parse_args()

    # check env
    conn_key = 'AZURE_STORAGE_CONNECTION_STRING'
    if conn_key not in os.environ:
        raise OSError(f'env var {conn_key} missing: cannot connect to azure')

    # auto clean-up
    try:

        # load in assets
        containers = ['assets', 'blend']
        for container in containers:
            dest = join('/', 'tmp', container)
            Path(dest).mkdir(exist_ok=True)
            run(
                [
                    'az',
                    'storage',
                    'blob',
                    'download-batch',
                    '--source',
                    container,
                    '--destination',
                    dest,
                    '--overwrite',
                ],
                capture_output=True,
                check=True,
            )

        # write file
        print(args.fname, file=open(args.fname, 'w'))

        # upload result
        run(
            [
                'az',
                'storage',
                'blob',
                'upload',
                '-f',
                args.fname,
                '-c',
                'rendered',
                '-n',
                args.fname,
            ],
            capture_output=True,
            check=True,
        )

    # delete temporary files
    finally:
        remove(args.fname)
        for container in containers:
            rmtree(join('/', 'tmp', container))
