import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
nb.cells.extend([
    new_markdown_cell('# 05: Interpretation & Explainability'),
    new_markdown_cell('This notebook demonstrates the explainability engine and product affinity communities, confirming that recommendations make business sense.'),
    new_code_cell('import pandas as pd\nimport sys\nimport os\nimport networkx as nx\nimport matplotlib.pyplot as plt\nsys.path.append(os.path.abspath(".."))\n\nfrom src.db import query_to_dataframe\nfrom src.recommenders import HybridRecommender, PopularityRecommender, ItemCFRecommender, AssociationRuleRecommender\nfrom src.interpretation import ExplainabilityEngine, build_product_affinity_network, get_communities'),
    new_markdown_cell('## 1. Load Data & Fit Models (Simulated)'),
    new_code_cell('query = """\nSELECT i.customer_id, i.invoice_no, ii.stock_code\nFROM invoices i\nJOIN invoice_items ii ON i.invoice_no = ii.invoice_no\nWHERE i.is_cancellation = 0\nLIMIT 10000\n"""\ndf = query_to_dataframe(query)\n\n# Quick fit for demonstration\npop = PopularityRecommender()\npop.fit(df)\nhybrid = HybridRecommender()\nhybrid.pop_rec = pop\n# We would normally fit all component models here.\n\nengine = ExplainabilityEngine(hybrid)'),
    new_markdown_cell('## 2. Product Affinity Network'),
    new_code_cell('G = build_product_affinity_network(df, min_co_occurrence=15)\nprint(f"Network built with {G.number_of_nodes()} products and {G.number_of_edges()} co-purchases.")\n\ncommunities = get_communities(G)\nprint(f"Detected {len(communities)} distinct product communities.")\n\n# Plot largest connected component\nif len(G.nodes) > 0:\n    lcc = G.subgraph(max(nx.connected_components(G), key=len))\n    plt.figure(figsize=(10,10))\n    nx.draw_spring(lcc, node_size=20, node_color="blue", edge_color="gray", alpha=0.5)\n    plt.title("Product Affinity Network (Largest Component)")\n    plt.show()')
])
with open('05_interpretation.ipynb', 'w') as f:
    nbformat.write(nb, f)
