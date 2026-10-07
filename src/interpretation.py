import networkx as nx
import pandas as pd
from itertools import combinations
from collections import defaultdict

class ExplainabilityEngine:
    def __init__(self, hybrid_recommender):
        self.recommender = hybrid_recommender
        
    def explain_recommendation(self, customer_id, stock_code, history):
        """
        Provides a human-readable justification for why an item was recommended.
        """
        explanations = []
        
        # Check Association Rules
        if self.recommender.rules_rec.rules is not None and not self.recommender.rules_rec.rules.empty:
            history_set = frozenset(history)
            rules = self.recommender.rules_rec.rules
            
            for _, row in rules.iterrows():
                if row['antecedents'].issubset(history_set) and stock_code in row['consequents']:
                    ants = list(row['antecedents'])
                    explanations.append(f"Customers who bought {ants} frequently co-purchased {stock_code}.")
                    break # Just need one strong rule explanation
                    
        # Check Popularity
        pop_items = self.recommender.pop_rec.popular_items
        if pop_items is not None:
            # If item is in top 50 globally
            top_50 = pop_items.head(50)['StockCode'].tolist()
            if stock_code in top_50:
                explanations.append(f"{stock_code} is currently a top-trending, highly popular item.")
                
        # Check Item-CF
        item_sim = self.recommender.item_rec.item_sim
        if item_sim is not None and stock_code in item_sim.columns:
            # Find the most similar item in history
            valid_hist = [item for item in history if item in item_sim.columns]
            if valid_hist:
                similarities = item_sim.loc[stock_code, valid_hist]
                most_similar = similarities.idxmax()
                max_sim = similarities.max()
                if max_sim > 0.1: # threshold
                    explanations.append(f"Recommended because it is very similar to {most_similar}, which you previously purchased.")
                    
        if not explanations:
            explanations.append("Recommended based on general purchasing patterns.")
            
        return " ".join(explanations)

def build_product_affinity_network(transactions, min_co_occurrence=10):
    """
    Builds a NetworkX graph of product co-purchases.
    transactions: DataFrame with InvoiceNo and StockCode
    """
    invoice_groups = transactions.groupby('InvoiceNo')['StockCode'].apply(list)
    
    co_occurrences = defaultdict(int)
    for items in invoice_groups:
        # Sort to avoid (A, B) and (B, A) being counted separately
        items = sorted(list(set(items)))
        for pair in combinations(items, 2):
            co_occurrences[pair] += 1
            
    G = nx.Graph()
    for (item1, item2), weight in co_occurrences.items():
        if weight >= min_co_occurrence:
            G.add_edge(item1, item2, weight=weight)
            
    return G

def get_communities(G):
    """
    Returns communities using greedy modularity.
    """
    return list(nx.community.greedy_modularity_communities(G))
