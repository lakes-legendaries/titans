"""Azure storage interface"""

import os
from os.path import basename, isdir
from subprocess import SubprocessError, run

from titans.sql import connect


def upload(
    source: str,
    /,
    *args,
    dest: str = None,
    server: str = 'dev',
):
    """Upload file or directory to website server

    Parameters
    ----------
    source: str
        file (or directory) to upload
    dest: str, optional, default=None
        destination file/folder. If None, use :code:`basename(source)`
    *args: Any
        extra args to pass to :code:`az storage blob upload` command
    server: str, optional, default='dev'
        server to upload to. Valid choices include :code:`prod` and
        :code:`dev`.
    """

    # check parameters
    if server not in ['prod', 'dev']:
        raise ValueError('Invalid server')

    # get credentials
    conn_str = connect().execute(f"""
        SELECT Value from creds
        Where Name = '{server}_conn'
    """).fetchone()[0]
    os.environ['AZURE_STORAGE_CONNECTION_STRING'] = conn_str

    # determine file or directory, and get destination
    is_dir = isdir(source)
    if dest is None:
        dest = basename(source)

    # build command
    cmd = [
        # command name
        'az',
        'storage',
        'blob',
        (
            'upload'
            if not is_dir
            else 'upload-batch'
        ),

        # container
        (
            '-c'
            if not is_dir
            else '-d'
        ),
        '$web',

        # destination
        (
            '-n'
            if not is_dir
            else '--destination-path'
        ),
        dest,

        # source
        (
            '-f'
            if not is_dir
            else '-s'
        ),
        source,

        # common flags
        '--overwrite',
    ]
    # add in extra flags
    cmd.extend(args)

    # execute upload command
    output = run(cmd, capture_output=True, check=True, text=True)
    if output.returncode:
        print('stdout:', output.stdout)
        print('stderr:', output.stderr)
        raise SubprocessError(f'Command {cmd} failed')


def release():
    """Publish dev server -> production"""

    # get credentials
    engine = connect()
    dev_sas = engine.execute("""
        SELECT Value from creds
        Where Name = 'dev_sas'
    """).fetchone()[0]
    prod_sas = engine.execute("""
        SELECT Value from creds
        Where Name = 'prod_sas'
    """).fetchone()[0]

    # get urls
    prefix_url = 'https://titansfileserver'
    suffix_url = '.blob.core.windows.net/$web/'
    dev_url = f'{prefix_url}dev{suffix_url}{dev_sas}'
    prod_url = f'{prefix_url}{suffix_url}{prod_sas}'

    # delete all blobs in production
    run(
        [
            'azcopy',
            'rm',
            prod_url,
            '--recursive',
            '--include-pattern',
            '*',
        ],
        capture_output=True,
        check=True,
    )

    # copy dev -> prod
    run(
        [
            'azcopy',
            'copy',
            dev_url,
            prod_url,
            '--recursive',
            '--include-pattern',
            '*',
        ],
        # capture_output=True,
        check=True,
    )
