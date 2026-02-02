# XGBoost-Imbalanced-Classification

**A machine learning pipeline optimized for binary classification on highly imbalanced datasets.**

## Overview
The goal of this project was to detect rare "Positive" events (Class 1) in a large, noisy dataset where the vast majority of samples are "Negative" (Class 0). 

Unlike standard classification tasks that prioritize Accuracy, this project focused on maximizing Partial AUC (pAUC) within a strict False Positive Rate range (max FPR = 0.01). The model was designed to prioritize high-precision detection, ensuring that when a positive prediction is made, it is highly likely to be correct.

## Key Results
The model was evaluated on a held-out validation set with the following performance metrics:

* **Partial AUC (max_fpr=0.01):** `0.9445` (Target Metric)
* **Precision (Class 1):** `1.00` (0 False Positives at optimized threshold)
* **Recall (Class 1):** `0.83`
* **F1-Score:** `0.91`

### Threshold Optimization
* **Optimized Threshold:** `0.8511` (Derived by maximizing F-Score on validation data)
* **Standard Threshold:** `0.5`
* **Impact:** The model is highly selective. On the final test set, the model identified **160** high-confidence positive instances using the optimized threshold.

## Technical Approach
1.  **Algorithm:** XGBoost Classifier (Extreme Gradient Boosting).
2.  **Handling Imbalance:** Applied `scale_pos_weight` calculated via the square root of the negative/positive ratio. This penalized False Negatives without over-correcting and causing False Positives.
3.  **Feature Selection:** Analyzed feature importance to remove 10 noisy features (e.g., `X6`, `X3`, `Time`), which improved model generalization.
4.  **Metric Selection:** Optimized for `aucpr` (Area Under the Precision-Recall Curve) during training to better suit the imbalanced nature of the data.
