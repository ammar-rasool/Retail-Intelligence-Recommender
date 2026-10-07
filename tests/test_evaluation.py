import pytest
import sys
import os
sys.path.append(os.path.abspath('..'))
from src.evaluation import recall_at_k, precision_at_k, hit_rate_at_k

def test_recall_at_k():
    recs = ['A', 'B', 'C']
    true = ['B', 'D']
    
    # K=2, recommended: A, B. intersection with true (B,D) is B. hits=1. true=2. recall = 0.5
    assert recall_at_k(recs, true, k=2) == 0.5
    
    # K=1, recommended: A. intersection is empty. recall = 0
    assert recall_at_k(recs, true, k=1) == 0.0

def test_precision_at_k():
    recs = ['A', 'B', 'C']
    true = ['B', 'D']
    
    # K=2, recommended: A, B. hits=1. precision = 1/2 = 0.5
    assert precision_at_k(recs, true, k=2) == 0.5
    
def test_hit_rate():
    recs = ['A', 'B', 'C']
    true = ['B', 'D']
    
    assert hit_rate_at_k(recs, true, k=3) == 1
    assert hit_rate_at_k(recs, ['Z'], k=3) == 0
