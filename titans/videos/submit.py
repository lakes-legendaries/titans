"""Submit batch jobs for rendering"""

from datetime import datetime
import json
import os
from os import remove
from os.path import isfile, join
import re
import sh

from titans.sql import connect


# cli
if __name__ == "__main__":

    # hard-coded parameters
    frames_per_job = 4

    # animation files details
    animations = {
        'No-Wait Anim': 1680,
    }

    # pull creds from db
    creds = {
        key: connect().execute(f"""
            SELECT Value from creds
            Where Name = '{key}'
        """).fetchone()[0]
        for key in ['azurecr', 'batch', 'prod_conn']
    }

    # set env vars that let us submit jobs to azure batch
    for key, value in {
        'AZURE_BATCH_ACCOUNT': 'titansbatch',
        'AZURE_BATCH_ENDPOINT': 'https://titansbatch.eastus.batch.azure.com',
        'AZURE_BATCH_ACCESS_KEY': creds['batch'],
    }.items():
        os.environ[key] = value

    # get json fname
    submission_time = re.sub(r'[:.]', r'-', datetime.now().isoformat())
    json_fname = join('/', 'tmp', f'{submission_time}.json')

    # submit task
    try:

        # render all frames
        for blender_fname, num_frames in animations.items():
            for first_frame in range(0, num_frames, frames_per_job):

                # get final frame
                final_frame = min(
                    first_frame + frames_per_job - 1,
                    num_frames - 1,
                )

                # create command
                az_env_var = 'AZURE_STORAGE_CONNECTION_STRING'
                cmd = re.sub(r'\s+', r' ', f"""/bin/bash -c '
                    docker login titansofeden.azurecr.io
                        --username titansofeden
                        --password "{creds['azurecr']}"
                    && docker run
                        --env {az_env_var}="{creds['prod_conn']}"
                        titansofeden.azurecr.io/titans:videos
                        --fname "{blender_fname}"
                        --first_frame {first_frame}
                        --final_frame {final_frame}
                '""")

                # create batch json
                task_json = [{
                    'id': (
                        'render'
                        f'-{submission_time}'
                        f"-{blender_fname.replace(' ', '_').lower()}"
                        f'-{first_frame}'
                    ),
                    'commandLine': cmd,
                    'userIdentity': {
                        'autoUser': {
                            'elevationLevel': 'admin'
                        }
                    },
                    'constraints': {
                        'maxTaskRetryCount': 3,
                    }
                }]
                json.dump(task_json, open(json_fname, 'w'))

                # create batch task
                sh.az(
                    'batch',
                    'task',
                    'create',
                    '--job-id',
                    'render',
                    '--json-file',
                    json_fname,
                )

    # clean-up
    finally:
        if isfile(json_fname):
            remove(json_fname)
