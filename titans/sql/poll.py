"""Create a table for a poll"""

from argparse import ArgumentParser

from titans.sql.common import connect


def create(table_name: str):
    """Create table for a poll

    Parameters
    ----------
    table_name: str
        name of sql table to create
    """
    connect().execute(f"""
        CREATE TABLE {table_name} (
            _ROWID_ INT NOT NULL AUTO_INCREMENT,
            Date DATETIME DEFAULT now(),
            Email VARCHAR(64) UNIQUE,
            Response VARCHAR(256),
            PRIMARY KEY (_ROWID_),
            KEY(Email)
        )
    """)


# command-line interface
if __name__ == '__main__':

    # parse cli
    parser = ArgumentParser(description='Create new poll table')
    parser.add_argument(
        'table_name',
        help=(
            'Name of table to create.'
            ' This must start with the suffix `poll-`.'
        ),
    )
    args = parser.parse_args()

    # make sure table starts with the suffix poll-
    if not args.table_name.startswith('poll_'):
        raise ValueError('table_name must start with the suffix `poll_`.')

    # create table
    create(table_name=args.table_name)
