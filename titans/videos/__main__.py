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


# create cli
app = typer.Typer()


def _submit_jobs(
        args: list[str],
        /,
        *,
        dependencies: list[list[int]] | None = None,
        local: bool = False,
):
    """Submit jobs to azure batch

    Parameters
    ----------
    args : list[str]
        command line arguments to pass to docker run command
    dependencies : list[str] | None, optional, default=None
        list of job names that must complete before this job can run
    local: bool, optional, default=False
        if True, run locally instead of submitting to azure batch
    """

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

    # function to name tasks
    submission_time = re.sub(r'[:.]', r'-', datetime.now().isoformat())
    def task_name(n: int, /) -> str:  # noqa
        return f"render-{submission_time}-{n}"

    # build tasks json
    all_tasks: list[dict] = []
    for a, argset in enumerate(args):

        # create command
        az_env_var = 'AZURE_STORAGE_CONNECTION_STRING'
        bash_cmd = re.sub(r'\s+', r' ', f"""
            docker login titansofeden.azurecr.io
                --username titansofeden
                --password "{creds['azurecr']}"
            && docker run
                --env {az_env_var}="{creds['prod_conn']}"
                titansofeden.azurecr.io/titans:videos
                {argset}
        """).strip()

        # create batch json
        all_tasks.append({
            'id': task_name(a),
            'commandLine': f"/bin/bash -c '{bash_cmd}'",
            'userIdentity': {
                'autoUser': {
                    'elevationLevel': 'admin',
                },
            },
            'constraints': {
                'maxTaskRetryCount': 3,
            },
            'dependsOn': {
                'taskIds': (
                    []
                    if dependencies is None
                    else [task_name(dep_num) for dep_num in dependencies[a]]
                ),
            },
        })

    # run locally
    if local:
        for task in all_tasks:
            bash_cmd = (
                task['commandLine']
                .replace('/bin/bash -c ', '')
                .replace('docker', 'sudo docker')
            )
            sh.bash('-c', bash_cmd)
        return

    # submit tasks
    for first_task in range(0, len(all_tasks), 100):  # 100 jobs per json

        # get task ceiling
        last_task = first_task + min(first_task + 100, len(all_tasks))

        # auto-clean-up writing json file
        try:

            # write json file
            json_fname = join('/', 'tmp', f'{submission_time}.json')
            with open(json_fname, 'w') as file:
                json.dump(all_tasks[first_task: last_task], file)

            # submit batch task
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


@app.command()
def animate(
    *,
    fname: str = Option(None, help="""
        if provided, only render this blender file (instead of all files). This
        should NOT contain the file extension.
    """),
    frame: int = Option(None, help="""
        if provided, only render this frame (for debugging). If you provide
        this, we strongly recommend you also provide fname.
    """),
    frames_per_job: int = Option(10, help="""
        Number of frames for each batch job. Fewer frames render faster, but
        have a higher marginal cost.
    """),
    local: bool = Option(False, help="""
        run locally (instead of on batch). For debugging.
    """),
):
    """Animate frames on azure batch"""

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

    # process args
    render_dict = (
        animations
        if not fname
        else {fname: animations[fname]}
    )

    # build argsets
    args = []
    for fname, num_frames in render_dict.items():
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

            # save argset
            args.append(f"""
                animate
                --fname "{fname}"
                --first_frame {first_frame}
                --final_frame {final_frame}
            """)

    # submit jobs
    _submit_jobs(args, local=local)


def _render(
        fname: str = None,
        local: bool = False,
):
    """Render videos on azure batch

    This function is called by render(), and provides an easy debugging
    interface (for when needed)
    """

    # render settings
    render_config: dict[str, dict[str, list[str] | bool]] = {
        "Card Flip": {
            "containers": [
                "assets",
                "blend",
            ],
            "dependencies": [],
            "mkv": True,
        },
        **{
            f"{element} Title": {
                "containers": [
                    "blend",
                    "overlays",
                ],
                "dependencies": [],
                "mkv": False,
            }
            for element in [
                "Storm",
                "Fire",
                "Ice",
                "Rock",
            ]
        },
        "Title": {
            "containers": [
                "blend",
                "overlays",
                "rendered",
            ],
            "dependencies": [
                "Storm Title",
                "Fire Title",
                "Ice Title",
                "Rock Title",
            ],
            "mkv": True,
        },
        "Title Video": {
            "containers": [
                "blend",
                "rendered",
            ],
            "dependencies": ["Title"],
            "mkv": True,
        },
        **{
            f"{name} Video": {
                "containers": [
                    "animated",
                    "blend",
                    "overlays",
                    "rendered",
                ],
                "dependencies": ["Title Video"],
                "mkv": True,
            }
            for name in [
                "Landing",
                "Empire",
                "No-Wait",
                "Constructed",
            ]
        },
    }

    # only do one file
    if fname is not None:
        render_config = {fname: render_config[fname]}
        render_config[fname]["dependencies"] = []

    # build argsets
    args = [
        f"""
            render
            --fname "{fname}"
            {"--mkv" if config["mkv"] else "--no-mkv"}
            {
                " ".join([
                    f"--containers {container}"
                    for container in config["containers"]
                ])
            }
        """
        for fname, config in render_config.items()
    ]

    # extract dependencies
    render_names = list(render_config.keys())
    dependency_names = [
        config["dependencies"]
        for config in render_config.values()
    ]
    dependency_nums = [
        [
            render_names.index(name)
            for name in dep_list
        ]
        for dep_list in dependency_names
    ]

    # submit jobs
    _submit_jobs(args, local=local, dependencies=dependency_nums)


@app.command()
def render(
    fname: str = Option(None, help="""
        if provided, only render this blender file (instead of all files). This
        should NOT contain the file extension.
    """),
    local: bool = Option(False, help="""
        run locally (instead of on batch). For debugging.
    """),
):
    """Render videos on azure batch"""
    _render(fname=fname, local=local)


# cli
if __name__ == "__main__":
    app()
