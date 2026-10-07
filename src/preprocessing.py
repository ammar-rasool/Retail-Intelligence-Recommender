import pandas as pd
import numpy as np

def audit_raw_data(df: pd.DataFrame) -> dict:
    """Audits the raw dataset to identify missing values, cancellations, etc."""
    audit = {}
    audit['total_records'] = len(df)
    audit['missing_customer_id'] = df['CustomerID'].isnull().sum()
    
    # Identify cancellations
    if 'InvoiceNo' in df.columns:
        cancellations = df[df['InvoiceNo'].astype(str).str.startswith('C')]
        audit['cancellation_records'] = len(cancellations)
    else:
        audit['cancellation_records'] = 0
        
    audit['negative_quantities'] = len(df[df['Quantity'] <= 0])
    
    return audit

def clean_data_for_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw dataset following the data cascade:
    Raw -> Drop Missing Customers -> Remove Cancellations -> Remove Negative Quantities
    """
    # 1. Drop missing CustomerIDs (guest checkout)
    clean_df = df.dropna(subset=['CustomerID']).copy()
    clean_df['CustomerID'] = clean_df['CustomerID'].astype(int)
    
    # 2. Filter out cancellations (InvoiceNo starts with 'C')
    clean_df = clean_df[~clean_df['InvoiceNo'].astype(str).str.startswith('C')]
    
    # 3. Filter out negative quantities or zero prices
    clean_df = clean_df[(clean_df['Quantity'] > 0) & (clean_df['UnitPrice'] > 0)]
    
    # 4. Standardize dates
    clean_df['InvoiceDate'] = pd.to_datetime(clean_df['InvoiceDate'])
    
    return clean_df

def prepare_tables_for_db(clean_df: pd.DataFrame):
    """
    Splits the cleaned pandas DataFrame into normalized DataFrames 
    matching the SQLite schema (customers, products, invoices, invoice_items).
    Returns a dictionary of DataFrames.
    """
    # Customers
    customers = clean_df[['CustomerID', 'Country']].drop_duplicates(subset=['CustomerID'])
    customers = customers.rename(columns={'CustomerID': 'customer_id', 'Country': 'country'})
    
    # Products
    products = clean_df[['StockCode', 'Description']].drop_duplicates(subset=['StockCode'])
    # Some stock codes have multiple descriptions over time, keep the first one
    products = products.groupby('StockCode', as_index=False).first()
    products = products.rename(columns={'StockCode': 'stock_code', 'Description': 'description'})
    
    # Invoices
    invoices = clean_df[['InvoiceNo', 'CustomerID', 'InvoiceDate']].drop_duplicates(subset=['InvoiceNo'])
    invoices = invoices.rename(columns={
        'InvoiceNo': 'invoice_no', 
        'CustomerID': 'customer_id', 
        'InvoiceDate': 'invoice_date'
    })
    invoices['is_cancellation'] = 0
    
    # Invoice Items
    invoice_items = clean_df[['InvoiceNo', 'StockCode', 'Quantity', 'UnitPrice']].copy()
    invoice_items = invoice_items.rename(columns={
        'InvoiceNo': 'invoice_no',
        'StockCode': 'stock_code',
        'Quantity': 'quantity',
        'UnitPrice': 'unit_price'
    })
    
    return {
        'customers': customers,
        'products': products,
        'invoices': invoices,
        'invoice_items': invoice_items
    }
