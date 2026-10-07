import sqlite3
import pandas as pd
import os

DB_PATH = 'retail_intelligence.db'

def get_connection():
    """Returns a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH)

def initialize_database(schema_path='../sql/schema.sql'):
    """Executes the schema script to initialize the database."""
    conn = get_connection()
    with open(schema_path, 'r') as f:
        schema = f.read()
    conn.executescript(schema)
    conn.commit()
    conn.close()

def query_to_dataframe(query, params=None):
    """Executes a SELECT query and returns a pandas DataFrame."""
    conn = get_connection()
    if params is None:
        df = pd.read_sql_query(query, conn)
    else:
        df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def load_dataframe_to_table(df, table_name, if_exists='append'):
    """Loads a pandas DataFrame into a SQLite table."""
    conn = get_connection()
    df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    conn.commit()
    conn.close()
