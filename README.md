# Retail Intelligence & Recommender Engine

This repository represents an empirical research study of transactional recommendation methods on retail data. It evaluates different recommendation strategies under realistic temporal evaluations, testing how their effectiveness varies with customer history.

## Research Questions

- **Central RQ**: *How do different transactional recommendation strategies perform under realistic temporal evaluation, and how does their effectiveness vary with customer history?*
  - **RQ1**: Does personalized recommendation outperform a time-aware popularity baseline?
  - **RQ2**: Which transactional approach performs best: Item-based Collaborative Filtering (Item-CF), User-based Collaborative Filtering (User-CF), or Association Rules?
  - **RQ3**: Does a validation-tuned hybrid model outperform its individual component scores?
  - **RQ4**: How does recommendation effectiveness change across customer history tiers (true cold-start vs. sparse vs. established)?
  - **RQ5**: What product communities and customer purchase behaviors explain recommendation successes and failures?

---

## Base Version Summary (Legacy Benchmark)

The initial version of this project (`DM Terminal Project.ipynb`) served as an exploratory data mining notebook. It included:
- **Exploratory Data Analysis**: Handling missing values, cleaning invoice fields, calculating standard RFM (Recency, Frequency, Monetary) metrics.
- **Similarity Analysis**: Computing pairwise Cosine similarity and Jaccard distance.
- **Clustering**: K-Means clustering on RFM attributes.
- **Association Rule Mining**: Apriori algorithm applied across the entire static dataset.
- **Classification**: Gaussian/Bernoulli Naive Bayes and Linear/RBF SVM models predicting Monetary RFM tiers (a circular target definition).

---

## Upgrade Transformation Matrix (New Architecture)

The repository has been entirely restructured from a monolithic notebook into a robust, 5-pillar Python architecture.

| Dimension | Base Version (`DM Terminal Project.ipynb`) | Upgraded Version (`retail-intelligence-recommender`) |
| :--- | :--- | :--- |
| **Core Goal** | Exploratory notebook & target classification | Empirical research study of transactional recommendation engines |
| **Data Architecture** | In-memory pandas DataFrames | Indexed SQLite database (`retail_intelligence.db`) with SQL queries |
| **Methodological Target** | Predict Monetary tier from RFM (circular) | Predict future customer purchases across temporal windows (zero leakage) |
| **Temporal Protocol** | Static full-dataset fit (future leakage) | Strict $T \le T_{\text{cutoff}}$ Train $\rightarrow$ Validation $\rightarrow$ Test temporal evaluation |
| **Recommender Suite** | Raw Apriori rule lookup | Time-Aware Popularity, Item-CF, User-CF, Assoc-Rules, Validation-Tuned Hybrid |
| **Cold-Start Evaluation** | None | Stratified history evaluation ($|H_u| = 0$, $1\text{--}2$, $3\text{--}5$, $>5$) |
| **Evaluation Metrics** | Classification accuracy / Precision | Ranking & Retrieval: **Recall@K**, **NDCG@K**, **Hit Rate@K**, **Catalog Coverage@K** |
| **Explainability** | Static rule output | Automated natural language recommendation justification engine |
| **Product Networks** | Basic co-occurrence | NetworkX community detection mapping customer history to product clusters |
| **Code Structure** | Single monolithic notebook | Modular 5-pillar Python package (`src/`), structured notebooks, Streamlit app |

---

## Data Pipeline & Experimental Protocol

1. **Data Quality Cascade**: Raw Records $\rightarrow$ Drop missing fields $\rightarrow$ Handle cancellations $\rightarrow$ Valid recommendation dataset.
2. **Feature Engineering**: Features are strictly separated into descriptive (for EDA) and model features (calculated strictly at $T_{\text{cutoff}}$).
3. **Temporal Split**:
   - **Train Window**: Generates component models (Item-CF, User-CF, Rules, Popularity).
   - **Validation Window**: Grid search tuning for Hybrid Recommender weights.
   - **Locked Test Window**: Final evaluation on untouched data.

---

## Running the Project

### Environment Setup
```bash
pip install -r requirements.txt
```

### Reproducing the Study
1. Run `notebooks/01_foundation.ipynb` to construct the SQLite database.
2. Run `notebooks/02_data_understanding.ipynb` for temporal and cohort EDA.
3. Run `notebooks/03_recommendation_models.ipynb` to fit and tune models.
4. Run `notebooks/04_evaluation.ipynb` to produce the final ablation study and cold-start results.
5. Run `notebooks/05_interpretation.ipynb` to view product communities.

### Launching the Dashboard
```bash
streamlit run dashboard/app.py
```
