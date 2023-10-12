"""Create a table for a poll"""

from argparse import ArgumentParser

import pandas as pd

from titans.sql.common import connect


# command-line interface
if __name__ == "__main__":
    # parse cli
    parser = ArgumentParser(description="Create new poll table")
    parser.add_argument(
        "table_name",
        help=(
            "Name of table to create."
            " This must start with the suffix `poll-`."
        ),
    )
    parser.add_argument(
        "--create",
        default=False,
        action="store_true",
        help="Create table",
    )
    parser.add_argument(
        "--list",
        default=False,
        action="store_true",
        help="List results",
    )
    args = parser.parse_args()

    # make sure table starts with the suffix poll-
    if not args.table_name.startswith("poll_"):
        raise ValueError("table_name must start with the suffix `poll_`.")

    # make sure only one action is specified
    if args.create and args.list:
        raise ValueError("Only one action can be specified.")

    # create table
    if args.create:
        connect().execute(
            f"""
            CREATE TABLE {args.table_name} (
                _ROWID_ INT NOT NULL AUTO_INCREMENT,
                Date DATETIME DEFAULT now(),
                Email VARCHAR(64) UNIQUE,
                Response VARCHAR(256),
                PRIMARY KEY (_ROWID_),
                KEY(Email)
            )
        """
        )

    # list results
    if args.list:
        for name, value in (
            pd.read_sql(
                f"SELECT Response FROM {args.table_name}",
                connect(),
            )
            .value_counts()
            .items()
        ):
            print(f"{value:3.0f}x: {name[0]:s}")
