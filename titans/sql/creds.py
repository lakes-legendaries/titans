"""Interface with email credentials table"""

from argparse import ArgumentParser
import json
import os
from os.path import join

from titans.sql.common import connect


def create(
    secrets_dir: str = os.environ['SECRETS_DIR'],
    email_creds_fname: str = 'titans-email-creds',
    email_token_fname: str = 'titans-email-token',
):
    """Create email credentials table

    Parameters
    ----------
    secrets_dir: str, optional, default=os.environ['SECRETS_DIR']
        secrets directory
    email_creds_fname: str, optional, default='titans-email-creds'
        file containing email credentials (TENANT, CLIENT_ID, and
        CLIENT_SECRET)
    email_token_fname: str, optional, default='titans-email-token'
        file containing email token
    """

    # load credentials and token
    token_data = json.load(open(join(secrets_dir, email_token_fname), 'r'))
    pairs = json.load(open(join(secrets_dir, email_creds_fname), 'r'))
    pairs['access_token'] = token_data['access_token']
    pairs['refresh_token'] = token_data['refresh_token']

    # create table
    engine = connect()
    engine.execute("""
        CREATE TABLE creds (
            Name TEXT,
            Value TEXT
        )
    """)

    # insert values
    for key, value in pairs.items():
        engine.execute(f"""
            INSERT INTO creds
            VALUES ("{key}", "{value}")
        """)


def delete():
    """Delete contacts table"""
    response = input(
        'WARNING: Email credentials table will be dropped. Continue? [y/n] '
    )
    if response == 'y':
        connect().execute('DROP TABLE creds')


# command-line interface
if __name__ == '__main__':

    # parse cli
    parser = ArgumentParser(description='Interface with MySQL database')
    parser.add_argument(
        '--create',
        default=False,
        action='store_true',
        help='Create email credentials table',
    )
    parser.add_argument(
        '--delete',
        default=False,
        action='store_true',
        help='Drop email credentials table',
    )
    args = parser.parse_args()

    # execute jobs
    if args.create:
        create()
    elif args.delete:
        delete()
