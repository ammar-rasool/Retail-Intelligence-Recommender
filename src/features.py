import pandas as pd
import numpy as np
from datetime import datetime

def compute_model_features(df: pd.DataFrame, t_cutoff: str) -> pd.DataFrame:
    """
    Computes customer behavioral features STRICTLY using data before t_cutoff.
    Prevents future information leakage into the model.
    
    Args:
        df: The raw transactions dataframe (with columns InvoiceNo, CustomerID, InvoiceDate, StockCode, Quantity, UnitPrice)
        t_cutoff: String or datetime representing the cutoff date (e.g., '2011-09-01')
        
    Returns:
        DataFrame with customer_id as index and behavioral features.
    """
    cutoff = pd.to_datetime(t_cutoff)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Strictly filter history before cutoff
    history = df[df['InvoiceDate'] < cutoff].copy()
    history['TotalRevenue'] = history['Quantity'] * history['UnitPrice']
    
    if history.empty:
        return pd.DataFrame()
        
    # Aggregate at Customer level
    agg_funcs = {
        'InvoiceNo': pd.Series.nunique,       # Total orders (Frequency)
        'TotalRevenue': 'sum',                # Total Spend (Monetary)
        'StockCode': pd.Series.nunique,       # Unique Products
        'Quantity': 'sum',                    # Total Items
        'InvoiceDate': ['min', 'max']         # First and Last purchase date in history
    }
    
    features = history.groupby('CustomerID').agg(agg_funcs)
    features.columns = ['TotalOrders', 'TotalRevenue', 'UniqueProducts', 'TotalItems', 'FirstPurchase', 'LastPurchase']
    
    # Calculate derived features
    features['AOV'] = features['TotalRevenue'] / features['TotalOrders']
    features['ProductDiversity'] = features['UniqueProducts'] / features['TotalItems']
    
    # Lifetime & Frequency
    features['CustomerLifetimeDays'] = (features['LastPurchase'] - features['FirstPurchase']).dt.days
    # Avoid division by zero for single-day purchasers
    features['CustomerLifetimeDays'] = features['CustomerLifetimeDays'].replace(0, 1)
    
    features['PurchaseFrequency'] = features['TotalOrders'] / (features['CustomerLifetimeDays'] / 30.0) # Orders per month
    
    # Recency as of cutoff
    features['RecencyDays'] = (cutoff - features['LastPurchase']).dt.days
    
    # Cleanup dates
    features = features.drop(columns=['FirstPurchase', 'LastPurchase'])
    
    return features

def compute_descriptive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes behavioral features using the full dataset. 
    WARNING: Use ONLY for Exploratory Data Analysis (EDA), not for predictive modeling!
    """
    # Simply use a future cutoff date to include everything
    future_date = pd.to_datetime(df['InvoiceDate'].max()) + pd.Timedelta(days=1)
    return compute_model_features(df, t_cutoff=future_date)
