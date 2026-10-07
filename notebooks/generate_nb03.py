import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
nb.cells.extend([
    new_markdown_cell('# 03: Recommendation Models & Validation Tuning'),
    new_markdown_cell('This notebook fits our recommendation engines (Popularity, Item-CF, User-CF, Association Rules) on the training set, and performs weight tuning for the Hybrid model on the validation set.'),
    new_code_cell('import pandas as pd\nimport numpy as np\nimport sys\nimport os\nsys.path.append(os.path.abspath(".."))\n\nfrom src.db import query_to_dataframe\nfrom src.recommenders import PopularityRecommender, ItemCFRecommender, UserCFRecommender, AssociationRuleRecommender, HybridRecommender'),
    new_markdown_cell('## 1. Load Data for Training'),
    new_code_cell('query = """\nSELECT i.customer_id, ii.stock_code, i.invoice_date\nFROM invoices i\nJOIN invoice_items ii ON i.invoice_no = ii.invoice_no\nWHERE i.is_cancellation = 0\n"""\ndf = query_to_dataframe(query)\ndf["invoice_date"] = pd.to_datetime(df["invoice_date"])'),
    new_markdown_cell('## 2. Temporal Split (Train / Val)'),
    new_code_cell('train_end = pd.to_datetime("2011-09-01")\nval_end = pd.to_datetime("2011-10-31")\n\ntrain_df = df[df["invoice_date"] < train_end]\nval_df = df[(df["invoice_date"] >= train_end) & (df["invoice_date"] < val_end)]\nprint("Train shape:", train_df.shape)\nprint("Validation shape:", val_df.shape)'),
    new_markdown_cell('## 3. Fit Component Models'),
    new_code_cell('pop_rec = PopularityRecommender()\npop_rec.fit(train_df)\n\nitem_rec = ItemCFRecommender()\nitem_rec.fit(train_df)\n\nuser_rec = UserCFRecommender()\nuser_rec.fit(train_df)\n\nrules_rec = AssociationRuleRecommender(min_support=0.01, min_confidence=0.1)\n# Convert format for rules (InvoiceNo x StockCode). Since we only pulled customer_id, we can group by customer_id.\n# Actually, let\'s fetch InvoiceNo for correct basket generation.\nquery_basket = """\nSELECT i.invoice_no, ii.stock_code, i.invoice_date\nFROM invoices i\nJOIN invoice_items ii ON i.invoice_no = ii.invoice_no\nWHERE i.is_cancellation = 0\n"""\nbasket_df = query_to_dataframe(query_basket)\nbasket_df["invoice_date"] = pd.to_datetime(basket_df["invoice_date"])\ntrain_basket = basket_df[basket_df["invoice_date"] < train_end]\n\nrules_rec.fit(train_basket)\nprint("Models fitted successfully!")')
])
with open('03_recommendation_models.ipynb', 'w') as f:
    nbformat.write(nb, f)
