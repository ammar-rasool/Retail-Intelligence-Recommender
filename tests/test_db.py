import pytest
import sqlite3
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath('..'))
from src.db import get_connection, initialize_database, load_dataframe_to_table

def test_db_connection():
    # Test with a dummy in-memory DB or default DB
    conn = get_connection()
    assert isinstance(conn, sqlite3.Connection)
    conn.close()

def test_table_loading():
    conn = get_connection()
    df = pd.DataFrame({'customer_id': [99999], 'country': ['Test']})
    load_dataframe_to_table(df, 'customers', if_exists='append')
    
    # Query back
    res = pd.read_sql_query("SELECT * FROM customers WHERE customer_id=99999", conn)
    assert not res.empty
    assert res.iloc[0]['country'] == 'Test'
    
    # Clean up
    conn.execute("DELETE FROM customers WHERE customer_id=99999")
    conn.commit()
    conn.close()
