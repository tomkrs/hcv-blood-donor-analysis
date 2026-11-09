# HCV Blood Donor Analysis: Anomaly Detection, Clustering & Classification

An end-to-end machine learning analysis of **laboratory blood values** to separate healthy blood donors from patients with **hepatitis C, fibrosis and cirrhosis**.

Machine learning course project, M.Sc. Medical Informatics, Heilbronn University (winter term 2023/24).

## Data

The [HCV dataset](https://archive.ics.uci.edu/dataset/571/hcv+data) from the UCI Machine Learning Repository (CC BY 4.0): 615 patients, age, sex and 10 lab values (ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, PROT). The course provided it in three stages:

| File | Used in | Content |
|---|---|---|
| `hcvdat11_Anomaly.csv` | Phase 1 | 100 samples, no labels |
| `hcvdat22_Clustering.csv` | Phase 2 | ~600 samples with missing values, no labels |
| `hcvdat33_Complete.csv` | Phase 3 | full data with diagnosis labels |

## Approach

**Phase 1: Unsupervised anomaly detection**
- Standardisation and encoding, then **PCA** (explained variance analysis)
- **Isolation Forest** vs. **Local Outlier Factor** on the PCA space. They flag 11 vs. 2 of 100 samples as anomalous, which shows how sensitive the results are to the choice of method.

**Phase 2: Imputation and clustering**
- Missing values: for each column, mean, median, most-frequent, **KNN** and **Iterative** imputation are compared by MSE after masking 10 % of each column. Then the best imputer is chosen per feature.
- Check whether the Phase 1 PCA model still fits the larger dataset
- **K-Means** and **Agglomerative Clustering** compared against the anomaly detectors, with decision boundaries visualised in PC1/PC2 space

**Phase 3: Interpretable classification**
- Preprocessing in a scikit-learn `Pipeline` / `ColumnTransformer`
- **Decision Tree** classifier (depth 3) with **87 % accuracy** on a 30 % held-out test set
- Decision paths printed per patient (e.g. `CHE <= 54.15 -> BIL > 9.65 -> ALP > 31.00 -> Blood Donor`), so a clinician can follow each prediction
- **Custom estimator** combining the decision tree with an anomaly detector: high overall accuracy hides weak recall on the rare disease classes, so every prediction also gets an anomaly flag as a safety net
- Robustness check with deliberately injected fibrosis and cirrhosis samples

## Key takeaway

Overall accuracy is misleading on this imbalanced dataset: blood donors make up the large majority. Looking at results **per diagnosis group** shows the weaker detection of hepatitis, fibrosis and cirrhosis. Adding an anomaly detector makes these cases visible instead of silently misclassifying them.

## Running it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter lab hcv_analysis.ipynb
```

The notebook was run with Python 3.9. All outputs are saved in the notebook.

## Tech stack

Python · pandas · scikit-learn · Matplotlib · Seaborn · Jupyter
