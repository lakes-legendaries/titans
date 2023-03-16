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

    # hard-coded parameters
    frames_dict = {
        'No-Wait Anim': 1680,
    }
    frames_per_job = 4

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
    json_fname = join('/', 'tmp', f'{suffix}.json')

    # submit task
    try:

        # render all frames
        for blender_fname, num_frames in frames_dict.items():
            for first_frame in range(0, num_frames, frames_per_job):

                # get final frame
                final_frame = min(
                    first_frame + frames_per_job - 1,
                    num_frames - 1,
                )

                # create command
                cmd = (
                    '/bin/bash -c'
                    ' "docker login'
                    ' titansofeden.azurecr.io'
                    ' --username titansofeden'
                    f' --password {creds["azurecr"]}'
                    ' && docker run --env'
                    ' AZURE_STORAGE_CONNECTION_STRING='
                    f'\\"{creds["prod_conn"]}\\"'
                    ' titansofeden.azurecr.io/titans:videos'
                    f' --fname \\"{blender_fname}\\"'
                    f' --first_frame {first_frame}'
                    f' --final_frame {final_frame}'
                    '"'
                )

                # create batch json
                pro_fname = blender_fname.replace(' ', '_').lower()
                task_id = f'render-{suffix}-{pro_fname}-{first_frame}'
                task_json = [{
                    "id": task_id,
                    "commandLine": cmd,
                    "userIdentity": {
                        "autoUser": {
                            "elevationLevel": "admin"
                        }
                    },
                    'constraints': {
                        'maxTaskRetryCount': 3,
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
