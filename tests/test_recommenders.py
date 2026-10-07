import pytest
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath('..'))
from src.recommenders import PopularityRecommender, ItemCFRecommender

@pytest.fixture
def dummy_transactions():
    return pd.DataFrame({
        'CustomerID': [1, 1, 2, 2, 3],
        'StockCode': ['A', 'B', 'B', 'C', 'A']
    })

def test_popularity_recommender(dummy_transactions):
    rec = PopularityRecommender()
    rec.fit(dummy_transactions)
    
    res = rec.recommend(history=[], k=1)
    assert not res.empty
    # 'A' and 'B' both have count 2, either could be first.
    assert res.iloc[0]['StockCode'] in ['A', 'B']

def test_item_cf_recommender(dummy_transactions):
    rec = ItemCFRecommender()
    rec.fit(dummy_transactions)
    
    # Customer bought A, what's similar?
    res = rec.recommend(history=['A'], k=1)
    # Customer 1 bought A and B. So B should have some similarity to A.
    assert not res.empty
