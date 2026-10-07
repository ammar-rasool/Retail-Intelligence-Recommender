import numpy as np
import pandas as pd

def recall_at_k(recommended_items, true_items, k=10):
    """Calculates Recall@K."""
    if not true_items:
        return 0.0
    recommended_at_k = recommended_items[:k]
    hits = len(set(recommended_at_k).intersection(set(true_items)))
    return hits / len(set(true_items))

def precision_at_k(recommended_items, true_items, k=10):
    """Calculates Precision@K."""
    if not recommended_items:
        return 0.0
    recommended_at_k = recommended_items[:k]
    hits = len(set(recommended_at_k).intersection(set(true_items)))
    return hits / len(recommended_at_k)

def hit_rate_at_k(recommended_items, true_items, k=10):
    """Calculates Hit Rate@K (1 if at least one hit, else 0)."""
    recommended_at_k = recommended_items[:k]
    hits = len(set(recommended_at_k).intersection(set(true_items)))
    return 1 if hits > 0 else 0

def ndcg_at_k(recommended_items, true_items, k=10):
    """Calculates NDCG@K."""
    if not true_items or not recommended_items:
        return 0.0
        
    recommended_at_k = recommended_items[:k]
    dcg = 0.0
    for i, item in enumerate(recommended_at_k):
        if item in true_items:
            dcg += 1.0 / np.log2(i + 2) # i+2 because index is 0-based and rank is 1-based
            
    # Ideal DCG
    idcg = 0.0
    ideal_hits = min(len(true_items), k)
    for i in range(ideal_hits):
        idcg += 1.0 / np.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0

def coverage_at_k(all_recommended_items, total_catalog_size):
    """Calculates Catalog Coverage."""
    unique_recommended = set(all_recommended_items)
    return len(unique_recommended) / total_catalog_size if total_catalog_size > 0 else 0.0

def evaluate_model(model, train_df, eval_df, k=10):
    """
    Evaluates a recommender model using Recall, NDCG, Precision, Hit Rate and Coverage.
    """
    # Create evaluation ground truth
    ground_truth = eval_df.groupby('customer_id')['stock_code'].apply(list).to_dict()
    # Create history
    history = train_df.groupby('customer_id')['stock_code'].apply(set).to_dict()
    
    total_catalog = set(train_df['stock_code'].unique()).union(set(eval_df['stock_code'].unique()))
    
    metrics = {'recall': [], 'precision': [], 'hit_rate': [], 'ndcg': []}
    all_recs = []
    
    for customer_id, true_items in ground_truth.items():
        hist = list(history.get(customer_id, []))
        
        # Get recommendations
        # Some models require customer_id, some require history
        try:
            recs_df = model.recommend(customer_id=customer_id, history=hist, k=k)
        except TypeError:
            try:
                recs_df = model.recommend(history=hist, k=k)
            except TypeError:
                recs_df = model.recommend(customer_id=customer_id, k=k)
                
        recs = recs_df['StockCode'].tolist() if not recs_df.empty else []
        all_recs.extend(recs)
        
        metrics['recall'].append(recall_at_k(recs, true_items, k))
        metrics['precision'].append(precision_at_k(recs, true_items, k))
        metrics['hit_rate'].append(hit_rate_at_k(recs, true_items, k))
        metrics['ndcg'].append(ndcg_at_k(recs, true_items, k))
        
    summary = {
        f'Recall@{k}': np.mean(metrics['recall']),
        f'Precision@{k}': np.mean(metrics['precision']),
        f'HitRate@{k}': np.mean(metrics['hit_rate']),
        f'NDCG@{k}': np.mean(metrics['ndcg']),
        f'Coverage@{k}': coverage_at_k(all_recs, len(total_catalog))
    }
    
    return summary

def stratify_customers_by_history(train_df):
    """
    Bins customers into Cold-Start, Sparse, Low History, Established.
    """
    counts = train_df.groupby('customer_id')['stock_code'].nunique()
    
    strata = {
        'true_cold_start': [],  # Actually requires customers not in train_df
        'sparse': counts[(counts >= 1) & (counts <= 2)].index.tolist(),
        'low_history': counts[(counts >= 3) & (counts <= 5)].index.tolist(),
        'established': counts[counts > 5].index.tolist()
    }
    return strata
