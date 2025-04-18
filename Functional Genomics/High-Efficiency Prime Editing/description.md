## Overview

This study builds on the work of Deshpande et al. (2024) to develop predictive models for forecasting dropout outcomes in prime editing screens. By analyzing sequence and design features, we show that machine learning models, including Random Forest, Logistic Regression, and Gradient Boosting, can moderately predict dropout events based on features such as GC content, edit length, and epegRNA subtype. These predictive models offer a computational framework for prioritizing mutations with a higher likelihood of causing functional impacts.

## Task

- **Data Generation:**
  - Dataset from Deshpande et al. (2024), focusing on dropout outcomes in prime editing screens.
  - Key features: Edit length, GC content, epegRNA type.
  - Target variable: Dropout defined as a Z-score below -2 at day 28 (Z score d28 avg < -2).
  - Addressed class imbalance by upsampling dropout instances to match non-dropouts.

- **Model Training:**
  - Models trained: Random Forest, Logistic Regression, Gradient Boosting.
  - Training on a 70/30 train-test split of the balanced dataset.
  - Default hyperparameters for initial evaluation.
  
- **Evaluation:**
  - Evaluated using Precision, Recall, F1-score, ROC AUC, and confusion matrices.
  - Performance comparison visualized using a grouped bar chart.

## Models & Setup

- **Models:** Random Forest, Logistic Regression, Gradient Boosting.
  - Hyperparameters: Default for initial evaluation.
  
- **Dataset:** Dataset from Deshpande et al. (2024), containing essential features such as:
  - Edit length: length of the intended genomic edit.
  - GC content: calculated GC fraction of the protospacer sequence.
  - epegRNA type: one-hot encoded categorical subtype.

## Metrics

- **Precision, Recall, F1-score:** Metrics for each class.
- **ROC AUC:** Area Under the Curve.
- **Confusion matrices:** Numeric and visual confusion matrices for model evaluation.

## Results

### 3.1 Random Forest Classifier (Baseline)

| Class        | Precision | Recall | F1-score | Support |
|--------------|-----------|--------|----------|---------|
| False        | 0.56      | 0.72   | 0.63     | 22558   |
| True         | 0.61      | 0.44   | 0.51     | 22666   |
| **Accuracy** |           |        | 0.58     | **45224**|

**Table 1:** Random Forest classification performance.

| Pred No | Pred Yes |
|---------|----------|
| 16287   | 6271     |
| 12717   | 9949     |

**Table 2:** Random Forest Confusion Matrix (numeric format).

**Figure 1:** Visual representation of the confusion matrix for the Random Forest classifier. The heatmap highlights class imbalance and model bias, with false negatives (bottom left) indicating difficulty in identifying true dropout cases.

### 3.2 Logistic Regression

| Class        | Precision | Recall | F1-score | Support |
|--------------|-----------|--------|----------|---------|
| False        | 0.58      | 0.54   | 0.56     | 22558   |
| True         | 0.57      | 0.61   | 0.59     | 22666   |
| **Accuracy** |           |        | 0.57     | **45224**|

**Table 3:** Logistic Regression classification performance.

AUC: 0.604

### 3.3 Gradient Boosting

| Class        | Precision | Recall | F1-score | Support |
|--------------|-----------|--------|----------|---------|
| False        | 0.56      | 0.72   | 0.63     | 22558   |
| True         | 0.61      | 0.44   | 0.51     | 22666   |
| **Accuracy** |           |        | 0.58     | **45224**|

**Table 4:** Gradient Boosting classification performance.

AUC: 0.610

### 3.4 Performance Summary

**Figure 2:** Bar chart comparing Precision, Recall, F1, and AUC across models. Gradient Boosting and Random Forest show similar behavior, while Logistic Regression offers slightly better recall but lower AUC.

## Conclusion

This study demonstrates that dropout outcomes in high-throughput prime editing screens can be partially predicted using machine learning models with minimal sequence-derived features. Random Forest and Gradient Boosting models showed the highest F1 and AUC scores, while Logistic Regression provided a solid baseline. Despite some model limitations, these findings highlight the potential of using ML models for functional variant prioritization and scalable experimental design. Future work could explore integrating mutation type annotations, gene essentiality scores, and deep learning approaches to further enhance predictive performance.
