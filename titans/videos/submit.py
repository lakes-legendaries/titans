"""Submit batch jobs for rendering"""

from datetime import datetime
import json
import os
from os import remove
from os.path import isfile, join
import re
from subprocess import run

from titans.sql import connect


# cli
if __name__ == "__main__":

    # pull creds
    creds = {
        key: connect().execute(f"""
            SELECT Value from creds
            Where Name = '{key}'
        """).fetchone()[0]
        for key in ['azurecr', 'batch', 'prod_conn']
    }

    # set batch env vars
    for key, value in {
        'AZURE_BATCH_ACCOUNT': 'titansbatch',
        'AZURE_BATCH_ENDPOINT': 'https://titansbatch.eastus.batch.azure.com',
        'AZURE_BATCH_ACCESS_KEY': creds['batch'],
    }.items():
        os.environ[key] = value

    # get task & json fname
    suffix = re.sub(r'[:.]', r'-', datetime.now().isoformat())
    task_id = f'test-{suffix}'
    json_fname = join('/', 'tmp', f'{task_id}.json')

    # submit task
    try:

        # create command
        cmd = (
            '/bin/bash -c'
            ' "docker login'
            ' titansofeden.azurecr.io'
            ' --username titansofeden'
            f' --password {creds["azurecr"]}'
            ' && docker run --env'
            f' AZURE_STORAGE_CONNECTION_STRING=\\"{creds["prod_conn"]}\\"'
            ' titansofeden.azurecr.io/titans:videos'
            ' --fname x3.txt"'
        )

        # create batch json
        task_json = [{
            "id": task_id,
            "commandLine": cmd,
            "userIdentity": {
                "autoUser": {
                    "elevationLevel": "admin"
                }
            }
        }]
        json.dump(task_json, open(json_fname, 'w'))

        # create batch task
        run(
            [
                'az',
                'batch',
                'task',
                'create',
                '--job-id',
                'render',
                '--json-file',
                json_fname,
            ],
            capture_output=True,
            check=True,
        )

    # clean-up
    finally:
        if isfile(json_fname):
            remove(json_fname)
