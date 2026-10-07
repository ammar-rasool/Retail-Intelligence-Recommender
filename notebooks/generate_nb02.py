import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
nb.cells.extend([
    new_markdown_cell('# 02: Data Understanding - Temporal Trends & Segmentation'),
    new_markdown_cell('This notebook performs supporting exploratory data analysis (EDA). We explore the descriptive features, cancellation audits, temporal purchasing behavior (monthly trends, cohort retention), and simple RFM clustering.'),
    new_code_cell('import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport sys\nimport os\nsys.path.append(os.path.abspath(".."))\n\nfrom src.db import query_to_dataframe\n\nplt.style.use("seaborn-v0_8-darkgrid")'),
    new_markdown_cell('## 1. Temporal Analysis: Revenue & Orders'),
    new_code_cell('with open("../sql/analysis.sql", "r") as f:\n    sql_script = f.read()\n\nqueries = sql_script.split(";")\n# First query is Monthly Revenue & Order Volume\nmonthly_trend_query = queries[0].strip()\nprint(monthly_trend_query)\n\nmonthly_df = query_to_dataframe(monthly_trend_query)\ndisplay(monthly_df.head())\n\nfig, ax1 = plt.subplots(figsize=(10, 5))\nax2 = ax1.twinx()\nax1.bar(monthly_df["invoice_month"], monthly_df["total_orders"], color="g", alpha=0.6)\nax2.plot(monthly_df["invoice_month"], monthly_df["total_revenue"], color="b", marker="o")\nax1.set_xlabel("Month")\nax1.set_ylabel("Total Orders", color="g")\nax2.set_ylabel("Total Revenue", color="b")\nplt.title("Monthly Orders vs Revenue")\nplt.xticks(rotation=45)\nplt.show()'),
    new_markdown_cell('## 2. Cohort Retention Matrix'),
    new_code_cell('cohort_query = queries[1].strip()\ncohort_df = query_to_dataframe(cohort_query)\n\ncohort_pivot = cohort_df.pivot(index="cohort_month", columns="activity_month", values="active_customers")\n\n# Calculate percentages\ncohort_sizes = cohort_pivot.iloc[:, 0]\nretention_matrix = cohort_pivot.divide(cohort_sizes, axis=0)\n\nplt.figure(figsize=(12, 8))\nsns.heatmap(retention_matrix, annot=True, fmt=".0%", cmap="Blues")\nplt.title("Cohort Retention Matrix")\nplt.show()'),
    new_markdown_cell('## 3. Top Popular Products'),
    new_code_cell('top_products_query = queries[2].strip()\ntop_products_df = query_to_dataframe(top_products_query)\ndisplay(top_products_df.head(10))')
])
with open('02_data_understanding.ipynb', 'w') as f:
    nbformat.write(nb, f)
