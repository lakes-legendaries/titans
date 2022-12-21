"""Interface with contacts table"""

from argparse import ArgumentParser

import numpy as np
from sqlalchemy.exc import IntegrityError

from titans.sql.common import connect, sanitize


def create():
    """Create contacts table"""
    connect().execute("""
        CREATE TABLE contacts (
            _ROWID_ INT NOT NULL AUTO_INCREMENT,
            Date DATETIME DEFAULT now(),
            Email VARCHAR(64) UNIQUE,
            PRIMARY KEY (_ROWID_),
            KEY(Email)
        )
    """)


def delete():
    """Delete contacts table"""
    response = input(
        'WARNING: Contacts table will be dropped. Continue? [y/n] '
    )
    if response == 'y':
        connect().execute('DROP TABLE contacts')


def upload(fname: str):
    """Upload saved contacts

    Parameters
    ----------
    fname: str
        text file containing emails
    """
    engine = connect()
    data = open(fname, 'r').read().splitlines()
    for email in np.unique(data):

        # sanitize email
        email = sanitize(email).lower()

        # skip empty
        if not len(email):
            continue

        # try to insert, ignore if duplicate
        try:
            engine.execute(f"""
                INSERT INTO contacts (Email)
                VALUES ("{email}")
            """)
        except IntegrityError:
            pass


# command-line interface
if __name__ == '__main__':

    # parse cli
    parser = ArgumentParser(description='Interface with MySQL database')
    parser.add_argument(
        '--create',
        default=False,
        action='store_true',
        help='Create contacts table',
    )
    parser.add_argument(
        '--delete',
        default=False,
        action='store_true',
        help='Drop contacts table',
    )
    parser.add_argument(
        '--upload',
        default=None,
        help='Upload emails in file to contacts table',
    )
    args = parser.parse_args()

    # execute jobs
    if args.create:
        create()
    elif args.delete:
        delete()
    elif args.upload:
        upload(args.upload)
