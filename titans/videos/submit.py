"""Submit batch jobs for rendering"""

from datetime import datetime
import json
import os
from os import remove
from os.path import isfile, join
import re
import sh

import typer
from typer import Option

from titans.sql import connect


# blender file parameters
animations: dict[str, int] = {
    '60-Sec Classic': 540,
    '60-Sec Constructed': 540,
    '60-Sec Haunt': 480,
    '60-Sec No Wait': 720,
    '60-Sec Opening': 660,
    '60-Sec Subvert': 540,
    '60-Sec Temples': 780,
    'Constructed Anim': 1600,
    'Empire Anim': 8600,
    'No-Wait Anim': 1680,
}

# cli help
frames_per_job_help: str = """
    Number of frames for each batch job. Fewer frames render faster, but have a
    higher marginal cost.
"""
local_help: str = """
    run locally (instead of on batch). For debugging.
"""
blender_fname_help: str = """
    if provided, only render this blender file (instead of all files)
"""
frame_help: str = """
    if provided, only render this frame (for debugging). If you provide this,
    we strongly recommend you also provide blender_fname.
"""

# cli function
def animate(
    frames_per_job: int = Option(4, help=frames_per_job_help),
    *,
    blender_fname: str = Option(None, help=blender_fname_help),
    local: bool = Option(False, help=local_help),
    frame: int = Option(None, help=frame_help),
):
    """Animate frames on azure batch"""

    # pull creds from db
    creds = {
        key: connect().execute(f"""
            SELECT Value from creds
            Where Name = '{key}'
        """).fetchone()[0]
        for key in ['azurecr', 'batch', 'prod_conn']
    }

    # process cli
    render_dict = (
        animations
        if not blender_fname
        else {blender_fname: animations[blender_fname]}
    )

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

        # get all tasks
        all_tasks: list[dict] = []
        for blender_fname, num_frames in render_dict.items():
            for first_frame in (
                range(0, num_frames, frames_per_job)
                if frame is None
                else [frame]
            ):

                # get final frame
                final_frame = min(
                    first_frame + frames_per_job - 1,
                    num_frames - 1,
                ) if frame is None else frame

                # create command
                az_env_var = 'AZURE_STORAGE_CONNECTION_STRING'
                bash_cmd = re.sub(r'\s+', r' ', f"""
                    docker login titansofeden.azurecr.io
                        --username titansofeden
                        --password "{creds['azurecr']}"
                    && docker run
                        --env {az_env_var}="{creds['prod_conn']}"
                        titansofeden.azurecr.io/titans:videos
                        --fname "{blender_fname}"
                        --first_frame {first_frame}
                        --final_frame {final_frame}
                """).strip()

                # run locally
                if local:
                    sh.bash('-c', bash_cmd.replace('docker', 'sudo docker'))
                    continue

                # create batch json
                all_tasks.append({
                    'id': (
                        'render'
                        f'-{submission_time}'
                        f"-{blender_fname.replace(' ', '_').lower()}"
                        f'-{first_frame}'
                    ),
                    'commandLine': f"/bin/bash -c '{bash_cmd}'",
                    'userIdentity': {
                        'autoUser': {
                            'elevationLevel': 'admin'
                        }
                    },
                    'constraints': {
                        'maxTaskRetryCount': 3,
                    }
                })

        # submit jobs to batch
        for first_task in range(0, len(all_tasks), 100):  # 100 jobs per json

            # get task ceiling
            last_task = first_task + min(first_task + 100, len(all_tasks))

            # create json
            with open(json_fname, 'w') as file:
                json.dump(all_tasks[first_task: last_task], file)

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


# cli
if __name__ == "__main__":
    typer.run(animate)
