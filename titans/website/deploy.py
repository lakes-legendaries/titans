"""Deploy to website servers"""

from argparse import ArgumentParser
from os import chdir, listdir
from os.path import dirname, join, realpath

from titans.cloud import release, upload
from titans.website.compile import compile


if __name__ == "__main__":
    # parse cli
    parser = ArgumentParser(description="Deploy assets to website")
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Deploy to production (default is deploy to dev)",
    )
    args = parser.parse_args()

    # deploy to prod
    if args.prod:
        release()

    # deploy to dev
    else:
        # compile files
        compile()

        # operate in this file's directory
        chdir(dirname(realpath(__file__)))

        # upload each html file
        folder = "site"
        for fname in listdir(folder):
            upload(join(folder, fname), "--content-type", "text/html")

        # upload supporting folders
        upload("script")
        upload("style")
