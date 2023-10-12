"""Interface with credentials table"""

from argparse import ArgumentParser
import json
import os
from os.path import join

from titans.sql.common import connect


def create(
    secrets_dir: str = os.environ["SECRETS_DIR"],
    batch_fname: str = "titans-batch",
    container_fname: str = "titans-container-registry",
    email_creds_fname: str = "titans-email-creds",
    email_token_fname: str = "titans-email-token",
    fileserver_dev_fname: str = "titans-fileserver-dev",
    fileserver_dev_sas_fname: str = "titans-fileserver-dev-sas",
    fileserver_fname: str = "titans-fileserver",
    fileserver_sas_fname: str = "titans-fileserver-sas",
):
    """Create email credentials table

    Parameters
    ----------
    secrets_dir: str, optional, default=os.environ['SECRETS_DIR']
        secrets directory
    batch_fname: str, optional, default='titans-batch'
        primary access key for azure batch account
    container_fname: str, optional, default='titans-container-registry'
        password for azure container registry
    email_creds_fname: str, optional, default='titans-email-creds'
        file containing email credentials (TENANT, CLIENT_ID, and
        CLIENT_SECRET)
    email_token_fname: str, optional, default='titans-email-token'
        file containing email token
    fileserver_dev_fname: str, optional, default='titans-fileserver-dev'
        azure connection string to dev fileserver
    fileserver_dev_sas_fname: str, optional, default='titans-fileserver-sas-dev'
        azcopy sas token for dev fileserver
    fileserver_fname: str, optional, default='titans-fileserver-dev'
        azure connection string to production fileserver
    fileserver_sas_fname: str, optional, default='titans-fileserver-sas'
        azcopy sas token for production fileserver
    """  # noqa

    # load credentials and token
    pairs = {}
    token_data = json.load(open(join(secrets_dir, email_token_fname), "r"))
    pairs = json.load(open(join(secrets_dir, email_creds_fname), "r"))
    pairs["access_token"] = token_data["access_token"]
    pairs["refresh_token"] = token_data["refresh_token"]
    pairs["dev_conn"] = (
        open(join(secrets_dir, fileserver_dev_fname), "r").read().strip()
    )
    pairs["dev_sas"] = (
        open(join(secrets_dir, fileserver_dev_sas_fname), "r").read().strip()
    )
    pairs["prod_conn"] = (
        open(join(secrets_dir, fileserver_fname), "r").read().strip()
    )
    pairs["prod_sas"] = (
        open(join(secrets_dir, fileserver_sas_fname), "r").read().strip()
    )
    pairs["azurecr"] = (
        open(join(secrets_dir, container_fname), "r").read().strip()
    )
    pairs["batch"] = open(join(secrets_dir, batch_fname), "r").read().strip()

    # create table
    engine = connect()
    engine.execute(
        """
        CREATE TABLE IF NOT EXISTS creds (
            Name TEXT,
            Value TEXT
        )
    """
    )

    # insert values
    for key, value in pairs.items():
        engine.execute(
            f"""
            INSERT INTO creds
            VALUES ("{key}", "{value.replace('%', '%%')}")
        """
        )


def delete():
    """Delete credentials table"""
    response = input(
        "WARNING: Credentials table will be dropped. Continue? [y/n] "
    )
    if response == "y":
        connect().execute("DROP TABLE creds")


# command-line interface
if __name__ == "__main__":
    # parse cli
    parser = ArgumentParser(
        description="Interface with MySQL credentials table"
    )
    parser.add_argument(
        "--create",
        default=False,
        action="store_true",
        help="Create email credentials table",
    )
    parser.add_argument(
        "--delete",
        default=False,
        action="store_true",
        help="Drop email credentials table",
    )
    args = parser.parse_args()

    # execute jobs
    if args.create:
        create()
    elif args.delete:
        delete()
