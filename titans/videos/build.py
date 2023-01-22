from os.path import join
from shutil import rmtree
from subprocess import run

from titans.cloud.files import download
from titans.sql import connect


# pull docker creds
docker_password = connect().execute("""
    SELECT Value from creds
    Where Name = 'azurecr'
""").fetchone()[0]

# do in loop for auto cleanup
odir = 'titans/videos'
containers = ['assets', 'blend']
try:
    # download assets and blend files
    for container in containers:
        download(container, join(odir, container))

    # build docker image
    run(
        [
            'sudo',
            'docker',
            'build',
            'titans/videos/',
            '-t',
            'titans:videos'
        ],
        capture_output=True,
        check=True,
    )

# clean up
finally:
    for container in containers:
        rmtree(join(odir, container))

# tag docekr image with registry name
run(
    [
        'sudo',
        'docker',
        'tag',
        'titans:videos',
        'titansofeden.azurecr.io/titans:videos',
    ],
    capture_output=True,
    check=True,
)

# login to docker registry
run(
    [
        'sudo',
        'docker',
        'login',
        'titansofeden.azurecr.io',
        '--username',
        'titansofeden',
        '--password',
        docker_password,
    ],
    capture_output=True,
    check=True,
)

# upload image
run(
    [
        'sudo',
        'docker',
        'push',
        'titansofeden.azurecr.io/titans:videos',
    ],
    capture_output=True,
    check=True,
)
