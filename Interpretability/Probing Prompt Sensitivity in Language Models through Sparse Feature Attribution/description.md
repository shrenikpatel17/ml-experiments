## Overview

This report investigates how semantically similar prompts activate internal representations differently within large language models. Inspired by Goodfire AI’s blog post [*Understanding and Steering LLaMA 3*](https://www.goodfire.ai/papers/understanding-and-steering-llama-3), we implement a full pipeline using **Mistral-7B-Instruct-v0.2** and a **Sparse Autoencoder (SAE)** to analyze how subtle prompt variations affect model internals. Using attribution techniques over mid-layer hidden states, we measure feature importance and representation overlap using similarity metrics. The analysis reveals how prompt phrasing steers LLM computation in interpretable, quantifiable ways.

## Task

- **Prompt Pair Generation:**
  - 15 semantically equivalent prompt pairs were manually crafted to vary in tone and structure while preserving intent.
  - Prompts covered a range of Python programming tasks.

- **Feature Attribution Pipeline:**
  - Extracted hidden states from transformer layer 16 of Mistral-7B.
  - Passed states through a trained Sparse Autoencoder with 1024 bottleneck units.
  - Computed the gradient of the first predicted token’s logit with respect to each SAE feature.
  - Selected the top-50 features per prompt using gradient magnitude as importance.

- **Similarity Evaluation:**
  - Compared prompt pair features using Jaccard Similarity and Cosine Similarity.
  - Visualized distributions and overlaps of key activations.

## Models & Setup

- **Model:** Mistral-7B-Instruct-v0.2
  - Locally run and used for forward passes and gradient attribution.

- **Sparse Autoencoder:**
  - 1024-feature bottleneck trained on mid-layer activations.
  - Trained over 3 epochs, using MSE loss with $\ell_1$ regularization.

- **Prompt Dataset:** 15 semantically equivalent prompt pairs.
  - Designed to test linguistic sensitivity while holding semantic intent constant.

## Metrics

- **Jaccard Similarity:** Measures overlap between top-50 features per prompt.
- **Cosine Similarity:** Measures alignment in attribution vectors across prompt pairs.
- **Token Consistency:** Tracks whether the same first token is predicted for each pair.

## Results

### 3.1 Summary Statistics

| Metric              | Value    |
|---------------------|----------|
| Avg. Jaccard        | 0.3516   |
| Avg. Cosine         | 0.8382   |
| Min Jaccard         | 0.0753   |
| Max Jaccard         | 0.7241   |

### 3.2 Insights

- Prompts with similar terminology (e.g., palindrome detection) had higher feature overlap.
- Prompts differing in task framing (e.g., "explain" vs. "show examples") showed lower overlap despite high cosine similarity.
- Almost all prompt pairs yielded the same predicted token, despite diverging internal activation patterns.

## Conclusion

This study supports the hypothesis that prompt engineering works by steering internal model activations. Even subtle linguistic changes activate different sparse features, despite consistent outputs. Jaccard similarity highlights the variability in “important” features, while cosine similarity suggests a stable underlying representation. The report confirms and extends the intuition from Goodfire’s blog, showing how interpretable tools like sparse autoencoders can demystify and measure LLM behavior. This pipeline opens the door to deeper interpretability, safety, and alignment analyses.