import sqlite3
from typing import Dict, Any, Union, Optional


class SQLLiteUtils:
    """
    A utility class for interacting with SQLite databases.

    This class allows you to connect to a SQLite database (or a remote database with a specified driver),
    check if a table exists, insert records into a table, and execute queries.

    Attributes:
    ----------
    database : str
        The name of the SQLite database.

    conn : sqlite3.Connection
        The connection object used to interact with the database.

    server : Optional[str], optional
        The server address for remote database connections, if applicable.

    driver : Optional[str], optional
        The driver used for remote database connections, if applicable.

    Methods:
    -------
    check_if_table_exists(table_name: str) -> bool:
        Checks if a given table exists in the database.

    insert_into_table_from_dictionary(table_name: str, data: Dict[str, Any]) -> None:
        Inserts a record into the specified table using the provided dictionary of data.

    select_from_table(query: str, how: str = 'many') -> Union[Dict[str, Any], list]:
        Executes a SELECT query on the database and returns the result.

    execute_query(query: str) -> None:
        Executes a non-SELECT query (e.g., INSERT, UPDATE, DELETE) on the database.
    """

    def __init__(self, database: str, is_private: bool = True, server: Optional[str] = None,
                 driver: Optional[str] = None):
        """
        Initializes the connection to the database.

        Parameters:
        ----------
        database : str
            The name of the SQLite database.

        is_private : bool, optional (default=True)
            If True, connects to a local SQLite database. If False, connects to a remote database.

        server : Optional[str], optional
            The server address for remote database connections, required if `is_private` is False.

        driver : Optional[str], optional
            The driver to use for remote database connections, required if `is_private` is False.
        """
        self.database = database

        if is_private:
            self.conn = sqlite3.connect(database, check_same_thread=False)
        else:
            self.server = server
            self.driver = driver
            self.conn = sqlite3.connect(f"""
                                        DRIVER={{{self.driver}}};
                                        SERVER={self.server};
                                        DATABASE={self.database};
                                        Trusted_Connection=yes;
                                        """, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def check_if_table_exists(self, table_name: str) -> bool:
        """
        Checks if the specified table exists in the database.

        Parameters:
        ----------
        table_name : str
            The name of the table to check.

        Returns:
        -------
        bool
            True if the table exists, False otherwise.
        """
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        table_exists = cursor.fetchone() is not None
        cursor.close()
        return table_exists

    def insert_into_table_from_dictionary(self, table_name: str, data: Dict[str, Any]) -> None:
        """
        Inserts a record into the specified table using the provided dictionary of data.

        Parameters:
        ----------
        table_name : str
            The name of the table to insert into.

        data : Dict[str, Any]
            A dictionary where the keys are column names and the values are the corresponding values to insert.
        """
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?'] * len(data))
        values = tuple(data.values())

        try:
            cursor = self.conn.cursor()
            query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            cursor.execute(query, values)
            self.conn.commit()
            cursor.close()
        except Exception as e:
            print(f"An error occurred while inserting record {data} into table {table_name}. Error: {e}")

    def select_from_table(self, query: str, how: str = 'many') -> Union[Dict[str, Any], list]:
        """
        Executes a SELECT query on the database and returns the result.

        Parameters:
        ----------
        query : str
            The SQL SELECT query to execute.

        how : str, optional (default='many')
            The format in which to return the results:
            - 'one' returns the first result as a dictionary.
            - 'many' returns all results as a list of dictionaries.

        Returns:
        -------
        Union[Dict[str, Any], list]
            The query result as a dictionary or list of dictionaries, depending on the `how` parameter.

        Raises:
        ------
        ValueError:
            If the `how` parameter is not 'one' or 'many'.
        """
        cursor = self.conn.cursor()
        cursor.execute(query)

        if how == 'one':
            result = cursor.fetchone()
            if result:
                return dict(result)
        elif how == 'many':
            result = cursor.fetchall()
            if result:
                return [dict(row) for row in result]
        else:
            raise ValueError("Invalid 'how' parameter. It must be one of ['many', 'one']")

        cursor.close()
        return result

    def execute_query(self, query: str) -> None:
        """
        Executes a non-SELECT query (e.g., INSERT, UPDATE, DELETE) on the database.

        Parameters:
        ----------
        query : str
            The SQL query to execute.
        """
        cursor = self.conn.cursor()
        cursor.execute(query)
        self.conn.commit()
        cursor.close()



