## Overview

This study enhances the **SPAC** (Self-Play with Adversarial Critic) method for offline preference optimization in large language model (LLM) alignment through detailed hyperparameter tuning. By performing a fine-grained grid search over λ and η values, we achieve a **31% accuracy improvement** over Supervised Fine-Tuning (SFT). The findings include a comprehensive evaluation using **DistilBERT** on a synthetic dataset, detailed visualizations, and insights into the role of learning rates in optimizing SPAC's performance.

## Task

- **Hyperparameter Tuning:**
  - Grid search over λ ∈ {0.5, 0.8, 1.0, 1.2, 1.5} and η ∈ {0.001, 0.01}.
  - Achieved 31% accuracy improvement over SFT.

- **Dataset:**
  - 500 synthetic prompt-response pairs for SPAC experiments, split into high and low coverage categories.
  - 80% training and 20% validation split.

- **Models & Implementation:**
  - Implemented both **SFT** and **SPAC** using **DistilBERT**.
  - SPAC uses a preference-based loss function optimized with hyperparameter tuning.

## Models & Setup

- **Model:** DistilBERT
  - Fine-tuned with synthetic dataset.
  - Training setup: Batch size of 8, 10 epochs, trained on CPU with 24GB RAM.

- **Hyperparameters:** 
  - λ ∈ {0.5, 0.8, 1.0, 1.2, 1.5}, η ∈ {0.001, 0.01}.
  - Evaluated performance for various λ and η values.

## Metrics

- **Accuracy:** Proportion of correctly predicted preferences.
- **Loss Analysis:** Examined training and validation loss for SPAC vs. SFT.

## Results

### 4.1 Initial Implementation

| Model            | Accuracy |
|------------------|----------|
| SFT              | 0.5700   |
| SPAC (λ = 1.0, η = 0.1) | 0.6200   |
| Improvement      | 5.00%    |

### 4.2 Enhanced Implementation

| Model            | η         | Accuracy |
|------------------|-----------|----------|
| SFT              | 0.0001    | 0.4300   |
| SPAC (λ = 1.0, η = 0.001) | 0.7400   |
| SPAC (λ = 1.5, η = 0.001) | 0.7400   |
| Improvement      | 31.00%   |

### 4.3 Learning Rate Impact

- A heatmap showed η = 0.01 resulted in stable accuracy (0.7400) across λ values, while η = 0.001 achieved peak performance at λ = 1.0 and 1.5.

### 4.4 Loss Analysis

- **SPAC Loss (λ = 1.0, η = 0.001):** Loss components (preference loss and regularization) balanced, with total loss stabilizing around 0.65.
- **SPAC vs. SFT Loss:** SPAC showed higher training loss but outperformed SFT in validation accuracy (0.73 vs. 0.58).

## Conclusion

Our results show that by refining SPAC’s hyperparameters, significant improvements in offline preference optimization are possible. The **31% accuracy gain** over SFT demonstrates SPAC’s potential for robust preference modeling. Future work could explore larger models and datasets, as well as deeper investigations into the threshold effect of learning rate on SPAC’s performance.
