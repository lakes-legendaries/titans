"""CLI for building and uploading docker image"""

from argparse import ArgumentParser
from subprocess import run

from titans.sql import connect


# run as cli
if __name__ == "__main__":
    # parse cli
    parser = ArgumentParser(description="create/upload docker image")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # pull docker creds
    docker_password = (
        connect()
        .execute(
            """
        SELECT Value from creds
        Where Name = 'azurecr'
    """
        )
        .fetchone()[0]
    )

    # build docker image
    run(
        ["sudo", "docker", "build", "titans/videos/", "-t", "titans:videos"],
        capture_output=not args.verbose,
        check=True,
    )

    # tag docekr image with registry name
    run(
        [
            "sudo",
            "docker",
            "tag",
            "titans:videos",
            "titansofeden.azurecr.io/titans:videos",
        ],
        capture_output=not args.verbose,
        check=True,
    )

    # login to docker registry
    run(
        [
            "sudo",
            "docker",
            "login",
            "titansofeden.azurecr.io",
            "--username",
            "titansofeden",
            "--password",
            docker_password,
        ],
        capture_output=not args.verbose,
        check=True,
    )

    # upload image
    run(
        [
            "sudo",
            "docker",
            "push",
            "titansofeden.azurecr.io/titans:videos",
        ],
        capture_output=not args.verbose,
        check=True,
    )
