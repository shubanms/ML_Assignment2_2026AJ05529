# ML Assignment 2 – Classification Models & Streamlit Deployment

**Name:** Shuban M S  **BITS ID:** 2026AJ05529  **Course:** Machine Learning, M.Tech AIML (WILP)

**Live app:** https://ml-assignment2-2026aj05529.streamlit.app/

## a. Problem statement
Online stores get many visitors, but only a small share of browsing sessions end in a purchase. The goal is to predict from a session's browsing behaviour (pages visited, time spent, bounce and exit rates, page value, month, visitor type) whether it will end in a purchase (`Revenue`). Five classification models are trained and compared on the same data, using metrics suited to an imbalanced target (about 15.5% purchases).

## b. Dataset description
| Item | Value |
|---|---|
| Source | UCI ML Repository – Online Shoppers Purchasing Intention Dataset (id 468): https://archive.ics.uci.edu/dataset/468 |
| Instances | 12,330 sessions (no missing values) |
| Features | 17 (15 numeric incl. `Weekend` as 0/1; 2 categorical: `Month`, `VisitorType`) |
| Target | `Revenue` – binary (True = session ended in a purchase) |
| Class balance | ~15.5% True / ~84.5% False (imbalanced) |
| Preprocessing | Booleans to 0/1; median imputation + standard scaling for numeric columns; most-frequent imputation + one-hot encoding for categorical columns; stratified 80/20 split (random_state=42) → 9,864 train / 2,466 test |

## c. GitHub repository
https://github.com/shubanms/ML_Assignment2_2026AJ05529

```
ML_Assignment2_2026AJ05529/
├── app.py                           # Streamlit app (loads saved models, no retraining)
├── ML_Assignment2_2026AJ05529.ipynb # training + evaluation notebook (run on BITS Virtual Lab)
├── requirements.txt
├── README.md
├── test_data.csv                    # held-out test split – upload this in the app
└── model/                           # 5 trained pipelines (.joblib), label encoder, metadata, metrics
```

The full dataset is downloaded inside the notebook with `ucimlrepo` (`fetch_ucirepo(id=468)`), so it is not stored in the repo.

## d. Models used

### Comparison table (test set, 2,466 rows)
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8812 | 0.8877 | 0.7432 | 0.3560 | 0.4814 | 0.4603 |
| Decision Tree | 0.8921 | 0.8770 | 0.6895 | 0.5524 | 0.6134 | 0.5562 |
| KNN | 0.8739 | 0.7993 | 0.6715 | 0.3639 | 0.4720 | 0.4322 |
| Naive Bayes | 0.6736 | 0.7939 | 0.2941 | 0.7906 | 0.4287 | 0.3249 |
| Support Vector Machine | 0.8861 | 0.8725 | 0.6996 | 0.4634 | 0.5575 | 0.5090 |

Precision, Recall and F1 are for the positive class (`Revenue = True`).

### Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest AUC (0.888) and the highest precision (0.743), so it ranks buyers well and its purchase predictions are usually right. But at the default 0.5 threshold it is too conservative: recall is only 0.356 (136 of 382 buyers found, 246 missed). Accuracy of 0.881 looks good but is barely above the 84.5% "always predict no purchase" baseline. Lowering the threshold would trade some precision for much better recall. |
| Decision Tree | Best overall on the metrics that matter for imbalanced data: highest accuracy (0.892), F1 (0.613) and MCC (0.556), finding 211 of 382 buyers with only 95 false alarms. The tree can split directly on strong signals like PageValues and capture non-linear thresholds that a linear model misses. max_depth=8 kept it from overfitting, although its probability estimates are coarser (AUC 0.877 vs 0.888 for LR). |
| KNN | Weakest of the distance/linear models: lowest AUC (0.799) apart from Naive Bayes, and recall of only 0.364 (243 buyers missed). With 15 scaled numeric features plus one-hot encoded Month and VisitorType, distances become less meaningful, and the 84.5% majority class dominates most neighbourhoods, pulling predictions towards "no purchase". |
| Naive Bayes | Behaves very differently from the others: highest recall (0.791, 302 of 382 buyers caught) but very low precision (0.294) with 725 false alarms, giving the lowest accuracy (0.674) and MCC (0.325). The feature-independence assumption is clearly violated here (e.g. page counts and page durations are strongly correlated), and the Gaussian assumption fits the heavily skewed duration and PageValues columns poorly, so its probabilities are badly calibrated. |
| Support Vector Machine | Consistently second-best: accuracy 0.886, AUC 0.873, F1 0.558 and MCC 0.509. The RBF kernel captures some non-linearity, so it finds more buyers than Logistic Regression (177 vs 136) while keeping false positives low (76). It was also the slowest model to train and its saved model is much larger than the linear and tree models. |
| **Overall winner for this dataset** | **Decision Tree.** Because only 15.5% of sessions end in a purchase, accuracy alone is misleading, so MCC and F1 are the fairest comparison, and the Decision Tree leads on both (MCC 0.556, F1 0.613) while also having the highest accuracy. If ranking quality (AUC) mattered most, Logistic Regression would be the choice; if catching every possible buyer mattered more than false alarms, Naive Bayes has the best recall. |
