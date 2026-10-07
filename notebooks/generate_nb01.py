import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
nb.cells.extend([
    new_markdown_cell('# 01: Foundation - Data Quality Cascade & Database Generation'),
    new_code_cell('import pandas as pd\nimport sqlite3\nimport sys\nimport os\nsys.path.append(os.path.abspath(".."))\n\nfrom src.preprocessing import audit_raw_data, clean_data_for_recommendations, prepare_tables_for_db\nfrom src.features import compute_descriptive_features\nfrom src.db import initialize_database, load_dataframe_to_table'),
    new_markdown_cell('## 1. Load Raw Data'),
    new_code_cell('df = pd.read_excel("../../Online Retail.xlsx")\nprint(df.shape)'),
    new_markdown_cell('## 2. Audit & Clean Data'),
    new_code_cell('audit = audit_raw_data(df)\nprint("Data Audit:", audit)\n\nclean_df = clean_data_for_recommendations(df)\nprint("Cleaned shape:", clean_df.shape)'),
    new_markdown_cell('## 3. Populate SQLite Database'),
    new_code_cell('initialize_database("../sql/schema.sql")\ntables = prepare_tables_for_db(clean_df)\nfor table_name, table_df in tables.items():\n    load_dataframe_to_table(table_df, table_name)\nprint("Database populated successfully!")')
])
with open('01_foundation.ipynb', 'w') as f:
    nbformat.write(nb, f)
