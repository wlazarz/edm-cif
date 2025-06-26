from utils.DatabaseUtils import SQLLiteUtils
from consts.queries import *

import os
import sys
from typing import Dict


def create_tables() -> None:
    """
    Creates necessary tables in the database if they do not exist. The table creation queries
    are executed in the specified order for each table name.

    This function uses the `SQLLiteUtils` class to interact with the database and ensure that
    each table exists. If a table does not exist, its creation query is executed.

    Returns:
    -------
    None
    """
    db_conn = SQLLiteUtils(database)

    tables_queries: Dict[str, str] = {
        datasets_table_name: create_datasets_table_query,
        outliers_table_name: create_outliers_table_query,
        contrastive_outlier_score_table_name: create_contrastive_outlier_score_query
    }

    for table_name, query in tables_queries.items():
        if not db_conn.check_if_table_exists(table_name):
            db_conn.execute_query(query)

    db_conn.conn.close()


def set_project_root() -> None:
    """
    Sets the root directory for the project and changes the working directory to it.
    This also updates the `sys.path` to include the root directory, ensuring that
    modules can be imported from the project root.

    Returns:
    -------
    None
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(project_root)
    sys.path.insert(0, project_root)

    print(f"Working Directory Set To: {os.getcwd()}")


def set_me_up() -> None:
    """
    Sets up the project by configuring the working directory and creating the necessary tables in the database.

    This function calls `set_project_root` to set the working directory and `create_tables` to ensure that all
    required database tables are present.

    Returns:
    -------
    None
    """
    set_project_root()
    create_tables()
