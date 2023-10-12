"""Shared functions"""

import os
import os.path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def connect(
    connection_str_fname: str = "titans-mysql",
) -> Engine:
    """Connect to MySQL database

    Parameters
    ----------
    database: str
        name of database to connect to
    connection_str_fname: str, optional, default='titans-mysql'
        filename containing sqlalchemy connection string (without database)

    Returns
    -------
    sqlalchemy.engine.Engine
        sqlalchemy connection engine
    """
    fname = os.path.join(os.environ["SECRETS_DIR"], connection_str_fname)
    connection_str = open(fname, "r").read().strip()
    return create_engine(connection_str)


def sanitize(input: str) -> str:
    """Sanitize input

    Parameters
    ----------
    input: str
        input value

    Returns
    -------
    str
        sanitized input
    """
    for original, replacement in [
        (r'"', r"`"),
        (r"'", r"`"),
        (r";", r","),
        (r"\\", r"/"),
    ]:
        input = input.replace(original, replacement)
    return input
