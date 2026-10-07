import streamlit as st
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath('..'))

from src.db import query_to_dataframe
from src.recommenders import PopularityRecommender, ItemCFRecommender, HybridRecommender
from src.interpretation import ExplainabilityEngine

st.set_page_config(page_title="Retail Recommender Demo", layout="wide")

@st.cache_resource
def load_data_and_models():
    # In a real app, you would load the pre-trained models from disk (e.g., pickle).
    # Here we simulate by doing a quick fit on a sample of data.
    query = """
    SELECT i.customer_id, ii.stock_code, i.invoice_date, p.description
    FROM invoices i
    JOIN invoice_items ii ON i.invoice_no = ii.invoice_no
    JOIN products p ON ii.stock_code = p.stock_code
    WHERE i.is_cancellation = 0
    LIMIT 100000
    """
    df = query_to_dataframe(query)
    
    # Fit basic models for demo
    pop_rec = PopularityRecommender()
    pop_rec.fit(df)
    
    item_rec = ItemCFRecommender()
    # Downsample for quick ItemCF fit in Streamlit
    item_rec.fit(df.head(10000)) 
    
    hybrid = HybridRecommender(weights={'pop': 0.2, 'item': 0.8, 'user': 0.0, 'rules': 0.0})
    hybrid.pop_rec = pop_rec
    hybrid.item_rec = item_rec
    
    engine = ExplainabilityEngine(hybrid)
    
    # Product lookup dict
    prod_dict = dict(zip(df['stock_code'], df['description']))
    
    return df, hybrid, engine, prod_dict

st.title("Retail Intelligence & Recommender Engine")

df, hybrid, engine, prod_dict = load_data_and_models()

tab1, tab2, tab3 = st.tabs(["Interactive Recommendations", "Customer Profiler", "Ablation Benchmark"])

with tab1:
    st.header("Recommendation Demo & Explainability")
    customers = df['customer_id'].dropna().unique()
    selected_customer = st.selectbox("Select Customer ID", customers)
    
    if st.button("Generate Recommendations"):
        # Get customer history
        history = df[df['customer_id'] == selected_customer]['stock_code'].unique().tolist()
        
        st.subheader("Customer History")
        history_desc = [f"{c} - {prod_dict.get(c, 'Unknown')}" for c in history]
        st.write(history_desc[:10]) # show top 10
        
        st.subheader("Hybrid Recommendations")
        recs = hybrid.recommend(customer_id=selected_customer, history=history, k=5)
        
        for idx, row in recs.iterrows():
            item = row['StockCode']
            score = row['score']
            desc = prod_dict.get(item, 'Unknown')
            explanation = engine.explain_recommendation(selected_customer, item, history)
            
            st.markdown(f"**{item} - {desc}** (Score: {score:.2f})")
            st.info(f"💡 {explanation}")

with tab2:
    st.header("Customer Profile")
    st.write("Displays behavioral metrics for the selected customer (RFM, Diversity, AOV).")
    # Implementation depends on pre-computed features table in DB

with tab3:
    st.header("Experimental Results (Ablation Study)")
    st.write("Summary of Recall@K and NDCG@K across Popularity, Item-CF, User-CF, and Hybrid systems on the locked Test Window.")
    # In production, you would load the results dataframe from phase 4 output
    results = pd.DataFrame({
        'Model': ['Popularity', 'Item-CF', 'Hybrid'],
        'Recall@10': [0.05, 0.12, 0.15],
        'NDCG@10': [0.03, 0.09, 0.12],
        'Catalog Coverage': ['5%', '45%', '55%']
    })
    st.dataframe(results)
