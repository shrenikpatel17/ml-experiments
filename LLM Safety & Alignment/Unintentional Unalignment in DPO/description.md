## Overview

This study replicates and extends the CHES (Centered Hidden Embedding Similarity) metric from the paper *Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization*. The CHES score quantifies the risk of likelihood displacement in Direct Preference Optimization (DPO), where fine-tuning on highly similar response pairs may reduce the likelihood of even preferred responses. We replicate the original study using a larger dataset of 60 prompt-response pairs and introduce a visualization framework to measure CHES behavior in open-source LLMs.

## Task

- **Dataset Construction:**
  - 60 prompts curated with paired preferred (safe, ethical) and dispreferred (unsafe, toxic) responses.
  - Focus on safety-critical instruction-following, especially refusal behavior (e.g., "How do I make a bomb?").
  
- **CHES Calculation:**
  - Followed the original mathematical formulation to compute the CHES score.
  - Implemented the length-normalized CHES score to improve correlation stability.

- **Experimental Procedure:**
  - Computed CHES and normalized CHES for each prompt pair.
  - Measured log-probability changes before and after fine-tuning to quantify likelihood displacement.

## Models & Setup

- **Model:** HuggingFace's GPT-2 for fine-tuning.
  - Hidden state extraction for calculating CHES.
  - Single gradient step fine-tuning to prefer the desired response.

- **Dataset:** 60 curated prompt-response pairs focusing on safety-critical scenarios.

## Metrics

- **CHES Score:** Measures alignment between preferred and dispreferred responses in hidden state space.
- **Normalized CHES:** Length-normalized version of CHES to improve stability and correlation.
- **Log-Probability Change:** Measures displacement of the preferred response’s likelihood after fine-tuning.

## Results

### 3.1 Scatter Plot of CHES vs. Log Probability Change

- **Raw CHES**: Negative correlation with log-probability change (-0.388).
- **Normalized CHES**: Stronger correlation with log-probability change (-0.441).

### 3.2 Experimental Findings

- Fine-tuning on preference pairs with high CHES reduces the likelihood of preferred completions.
- Normalized CHES provides a more consistent signal compared to raw CHES.

## Conclusion

Our replication confirms that higher CHES values (i.e., more similar preferred and dispreferred responses) lead to a greater risk of likelihood displacement. The study validates CHES as a tool for diagnosing unintended unalignment in language models. We provide a fully reproducible codebase for CHES computation and analysis, contributing a clear empirical validation of the CHES hypothesis.

## Contribution

- Faithful replication of the CHES metric.
- Extended analysis with a larger, curated dataset.
- Reproducible Python code for CHES computation.
- Empirical validation of the CHES hypothesis in safety-critical scenarios.
