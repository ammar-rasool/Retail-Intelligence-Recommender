import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.frequent_patterns import apriori, association_rules

class PopularityRecommender:
    def __init__(self):
        self.popular_items = None
        
    def fit(self, transactions: pd.DataFrame):
        """Fits on transactions (CustomerID, StockCode)."""
        item_counts = transactions.groupby('StockCode').size().reset_index(name='count')
        self.popular_items = item_counts.sort_values('count', ascending=False)
        
    def recommend(self, customer_id=None, k=10, history=None):
        """Returns top k popular items not in history."""
        recs = self.popular_items.copy()
        if history:
            recs = recs[~recs['StockCode'].isin(history)]
        
        # Max-normalize the scores so they are in [0, 1]
        max_count = recs['count'].max()
        recs['score'] = recs['count'] / max_count if max_count > 0 else 0
        return recs.head(k)[['StockCode', 'score']]

class ItemCFRecommender:
    def __init__(self):
        self.item_sim = None
        self.item_mapper = None
        
    def fit(self, transactions: pd.DataFrame):
        user_item_matrix = transactions.pivot_table(index='CustomerID', columns='StockCode', aggfunc='size', fill_value=0)
        user_item_matrix = (user_item_matrix > 0).astype(int)
        
        sim_matrix = cosine_similarity(user_item_matrix.T)
        self.item_sim = pd.DataFrame(sim_matrix, index=user_item_matrix.columns, columns=user_item_matrix.columns)
        
    def recommend(self, history, k=10):
        if not history or self.item_sim is None:
            return pd.DataFrame(columns=['StockCode', 'score'])
            
        valid_history = [item for item in history if item in self.item_sim.columns]
        if not valid_history:
            return pd.DataFrame(columns=['StockCode', 'score'])
            
        # Sum similarities across all items in history
        scores = self.item_sim[valid_history].sum(axis=1)
        scores = scores.drop(index=valid_history, errors='ignore')
        
        scores = scores.sort_values(ascending=False).head(k).reset_index()
        scores.columns = ['StockCode', 'score']
        
        # Normalize to [0, 1] using max possible similarity sum (len of history)
        if len(valid_history) > 0:
            scores['score'] = scores['score'] / len(valid_history)
            
        return scores

class UserCFRecommender:
    def __init__(self):
        self.user_item_matrix = None
        self.user_sim = None
        
    def fit(self, transactions: pd.DataFrame):
        self.user_item_matrix = transactions.pivot_table(index='CustomerID', columns='StockCode', aggfunc='size', fill_value=0)
        self.user_item_matrix = (self.user_item_matrix > 0).astype(int)
        
        sim_matrix = cosine_similarity(self.user_item_matrix)
        self.user_sim = pd.DataFrame(sim_matrix, index=self.user_item_matrix.index, columns=self.user_item_matrix.index)
        
    def recommend(self, customer_id, k=10, history=None):
        if self.user_sim is None or customer_id not in self.user_sim.index:
            return pd.DataFrame(columns=['StockCode', 'score'])
            
        # Get similar users
        sim_users = self.user_sim[customer_id].drop(index=customer_id).sort_values(ascending=False)
        sim_users = sim_users[sim_users > 0]
        
        if sim_users.empty:
            return pd.DataFrame(columns=['StockCode', 'score'])
            
        # Weighted sum of items bought by similar users
        neighbor_matrix = self.user_item_matrix.loc[sim_users.index]
        item_scores = neighbor_matrix.T.dot(sim_users)
        
        if history:
            item_scores = item_scores.drop(index=history, errors='ignore')
            
        scores = item_scores.sort_values(ascending=False).head(k).reset_index()
        scores.columns = ['StockCode', 'score']
        
        # Normalize
        max_score = scores['score'].max()
        if max_score > 0:
            scores['score'] = scores['score'] / max_score
            
        return scores

class AssociationRuleRecommender:
    def __init__(self, min_support=0.01, min_confidence=0.1):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.rules = None
        
    def fit(self, transactions: pd.DataFrame):
        # Create basket format (InvoiceNo x StockCode)
        basket = transactions.pivot_table(index='InvoiceNo', columns='StockCode', aggfunc='size', fill_value=0)
        basket = (basket > 0).astype(bool)
        
        frequent_itemsets = apriori(basket, min_support=self.min_support, use_colnames=True)
        if not frequent_itemsets.empty:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=self.min_confidence)
            self.rules = rules[['antecedents', 'consequents', 'lift', 'confidence', 'support']]
        else:
            self.rules = pd.DataFrame()
            
    def recommend(self, history, k=10):
        if self.rules is None or self.rules.empty or not history:
            return pd.DataFrame(columns=['StockCode', 'score'])
            
        history_set = frozenset(history)
        
        recs = {}
        for idx, row in self.rules.iterrows():
            antecedents = row['antecedents']
            consequents = row['consequents']
            
            # If all antecedents are in history
            if antecedents.issubset(history_set):
                for item in consequents:
                    if item not in history_set:
                        # Use lift as score
                        recs[item] = max(recs.get(item, 0), row['lift'])
                        
        if not recs:
            return pd.DataFrame(columns=['StockCode', 'score'])
            
        scores = pd.DataFrame(list(recs.items()), columns=['StockCode', 'score'])
        scores = scores.sort_values(ascending=False).head(k)
        
        # Normalize
        max_score = scores['score'].max()
        if max_score > 0:
            scores['score'] = scores['score'] / max_score
            
        return scores

class HybridRecommender:
    def __init__(self, weights={'pop': 0.1, 'item': 0.4, 'user': 0.1, 'rules': 0.4}):
        self.weights = weights
        self.pop_rec = PopularityRecommender()
        self.item_rec = ItemCFRecommender()
        self.user_rec = UserCFRecommender()
        self.rules_rec = AssociationRuleRecommender()
        
    def fit(self, transactions: pd.DataFrame):
        self.pop_rec.fit(transactions)
        self.item_rec.fit(transactions)
        self.user_rec.fit(transactions)
        self.rules_rec.fit(transactions)
        
    def recommend(self, customer_id, history, k=10):
        # Get raw normalized predictions
        pop_res = self.pop_rec.recommend(customer_id=customer_id, history=history, k=k*3)
        item_res = self.item_rec.recommend(history=history, k=k*3)
        user_res = self.user_rec.recommend(customer_id=customer_id, history=history, k=k*3)
        rules_res = self.rules_rec.recommend(history=history, k=k*3)
        
        # Combine
        combined = {}
        for item, score in zip(pop_res['StockCode'], pop_res['score']):
            combined[item] = combined.get(item, 0) + self.weights['pop'] * score
            
        for item, score in zip(item_res['StockCode'], item_res['score']):
            combined[item] = combined.get(item, 0) + self.weights['item'] * score
            
        for item, score in zip(user_res['StockCode'], user_res['score']):
            combined[item] = combined.get(item, 0) + self.weights['user'] * score
            
        for item, score in zip(rules_res['StockCode'], rules_res['score']):
            combined[item] = combined.get(item, 0) + self.weights['rules'] * score
            
        if not combined:
            return pd.DataFrame(columns=['StockCode', 'score'])
            
        res_df = pd.DataFrame(list(combined.items()), columns=['StockCode', 'score'])
        return res_df.sort_values('score', ascending=False).head(k)
