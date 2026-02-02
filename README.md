# XGBoost-Imbalanced-Classification

**A machine learning pipeline optimized for binary classification on a highly imbalanced dataset.**

## Overview
This project addresses the problem of detecting rare positive events in a large, noisy dataset where the majority of observations belong to the negative class.

Standard accuracy is not an appropriate objective in this setting. Instead, the model is optimized for partial AUC (pAUC) under a strict false positive rate constraint (max_fpr = 0.01). The goal is high precision detection, ensuring that predicted positives are highly reliable.

## Data Description

* `X1, X2, …, X29`: Numerical features describing each online session
* `Time`: Elapsed time since the first recorded observation
* `Label`: Binary response variable - `0` = Negative, `1` = Positive
* `id`: Record identifier used in `test.csv`

## Key Results
The model was evaluated on a held-out validation set with the following performance metrics:

* **Partial AUC (max_fpr=0.01):** `0.9445`
* **Precision (Class 1):** `1.00`
* **Recall (Class 1):** `0.83`
* **F1-Score:** `0.91`

At the optimized threshold, the model produced zero false positives on the validation set.

### Threshold Optimization
* **Optimized Threshold:** `0.8511` (Derived by maximizing F-Score on validation data)
* **Standard Threshold:** `0.5`

While the optimized threshold provided maximum precision on validation data, the standard threshold was selected for the final test set to ensure better model generalization and recall stability. On the final test set, the standard threshold identified 177 positive instances, capturing more potential events than the 160 identified by the stricter optimized threshold.

## Modeling Approach
1.  **Algorithm:** XGBoost Classifier (Extreme Gradient Boosting).
2.  **Class Imbalance Handling:** The `scale_pos_weight` parameter was set to the square root of the negative-to-positive class ratio. This penalizes false negatives without over-correcting toward false positives.
3.  **Feature Selection:** Feature importance analysis was used to remove noisy predictors, including X3, X6, and Time. This improved generalization on validation data.
4.  **Training Metric:** The model was trained using aucpr (Area Under the Precision-Recall Curve), which is more informative than ROC-AUC for highly imbalanced classification problems.

## Project Structure

* `finalmodel.ipynb` - Final cleaned notebook used to generate results
* `model.ipynb` – Development and experimentation notebook
* `submission.csv` – Final model predictions
* `requirements.txt` – Python dependencies
* `README.md` – Project documentation

## Dependencies
* pandas
* numpy
* matplotlib
* scikit-learn
* xgboost
* jupyter

## Notes and Limitations
* The model is intentionally conservative and may miss some true positives.
* Performance is optimized for low false positive rates and may not generalize to settings with different cost tradeoffs.

## Kaggle Competition Results
This project was developed for a Kaggle binary classification competition focused on extreme class imbalance and partial AUC evaluation.

* Public leaderboard:
    * Rank: 49 / 51
    * Score: 0.9469625

* Private leaderboard:
    * Rank: 13 / 51
    * Score: 0.9514578

The private leaderboard result reflects stronger generalization performance and aligns with the validation metrics reported above.
