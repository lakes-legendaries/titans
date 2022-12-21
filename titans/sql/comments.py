"""Interface with comments table"""

from argparse import ArgumentParser

from titans.sql.common import connect


def create():
    """Create contacts table"""
    connect().execute("""
        CREATE TABLE comments (
            _ROWID_ INT NOT NULL AUTO_INCREMENT,
            Date DATETIME DEFAULT now(),
            Email VARCHAR(64),
            Comment TEXT,
            PRIMARY KEY (_ROWID_),
            KEY(Email)
        )
    """)


def delete():
    """Delete contacts table"""
    response = input(
        'WARNING: Comments table will be dropped. Continue? [y/n] '
    )
    if response == 'y':
        connect().execute('DROP TABLE comments')


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
    args = parser.parse_args()

    # execute jobs
    if args.create:
        create()
    elif args.delete:
        delete()
