from os.path import join
from shutil import rmtree
from subprocess import run

from titans.cloud.files import download


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
        # capture_output=True,
        check=True,
    )

# clean up
finally:
    for container in containers:
        rmtree(join(odir, container))
