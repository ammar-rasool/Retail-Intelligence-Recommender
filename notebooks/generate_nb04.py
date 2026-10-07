import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
nb.cells.extend([
    new_markdown_cell('# 04: Experimental Evaluation (Locked Test Window)'),
    new_markdown_cell('This notebook evaluates the models on the untouched test set. We calculate Recall@K, NDCG@K, Hit Rate, and Coverage.'),
    new_code_cell('import pandas as pd\nimport numpy as np\nimport sys\nimport os\nsys.path.append(os.path.abspath(".."))\n\nfrom src.db import query_to_dataframe\nfrom src.recommenders import PopularityRecommender, ItemCFRecommender, UserCFRecommender, AssociationRuleRecommender, HybridRecommender\nfrom src.evaluation import evaluate_model, stratify_customers_by_history'),
    new_markdown_cell('## 1. Load Locked Test Data'),
    new_code_cell('query = """\nSELECT i.customer_id, ii.stock_code, i.invoice_date\nFROM invoices i\nJOIN invoice_items ii ON i.invoice_no = ii.invoice_no\nWHERE i.is_cancellation = 0\n"""\ndf = query_to_dataframe(query)\ndf["invoice_date"] = pd.to_datetime(df["invoice_date"])\n\ntrain_end = pd.to_datetime("2011-10-31") # Validation was up to Oct 31\n\ntrain_and_val_df = df[df["invoice_date"] < train_end]\ntest_df = df[df["invoice_date"] >= train_end]\nprint("Training + Val history shape:", train_and_val_df.shape)\nprint("Test (Ground Truth) shape:", test_df.shape)'),
    new_markdown_cell('## 2. Fit Models on Train+Val, Evaluate on Test'),
    new_code_cell('pop_rec = PopularityRecommender()\npop_rec.fit(train_and_val_df)\n\nprint("Evaluating Popularity Baseline...")\npop_metrics = evaluate_model(pop_rec, train_and_val_df, test_df, k=10)\nprint("Popularity Metrics:", pop_metrics)'),
    new_markdown_cell('## 3. History Stratification (Cold-Start Experiment)'),
    new_code_cell('strata = stratify_customers_by_history(train_and_val_df)\nprint("Strata counts:", {k: len(v) for k, v in strata.items()})\n\n# Evaluate specifically on the low_history cohort\nlow_history_test = test_df[test_df["customer_id"].isin(strata["low_history"])]\nprint("Evaluating Popularity on Low History cohort...")\npop_low_metrics = evaluate_model(pop_rec, train_and_val_df, low_history_test, k=10)\nprint("Low History Popularity Metrics:", pop_low_metrics)')
])
with open('04_evaluation.ipynb', 'w') as f:
    nbformat.write(nb, f)
