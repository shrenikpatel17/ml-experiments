## Overview

This experiment evaluates the **internal consistency** of a large language model (**Llama3.2**) under varying **inference-time compute budgets**. Inspired by the *Thinking Slow, Fast* framework, the study introduces **agreement ratio** as a novel diagnostic metric, analyzing how frequently multiple completions from the same prompt converge on the same answer.

## Task

Given 10 curated **GSM8K math word problems**, the model generates multiple completions per prompt:

- Completions are sampled with \( k \in \{1, 3, 5\} \).
- Final answers are parsed from each completion.
- The objective is to analyze:
  - **Final answer via majority vote**
  - **Agreement ratio** among completions
  - **Inference time** at each value of \( k \)

## Models & Setup

- **Model:** Llama3.2 with temperature sampling  
  - `temperature = 1.0`, `max_new_tokens = 1000`
- **Dataset:** 10 manually selected problems from GSM8K
- **Sampling:** Completions generated independently per prompt for each \( k \)

## Metrics

- **Mean agreement ratio** across problems
- **Number of unanimous completions**
- **Distribution of agreement counts per problem**
- **Inference time (wall-clock) vs. agreement tradeoff**

## Results

- **Agreement increased** from ~0.30 (k=1) to ~0.40 (k=5).
- **Inference time** scaled linearly: ~46s at k=1 to ~240s at k=5.
- Some problems remained unstable across all values of \( k \), indicating **problem sensitivity** or **prompt ambiguity**.

## Conclusion

The agreement ratio offers a lightweight, label-free diagnostic to assess **model confidence and reasoning consistency**. Unlike standard accuracy or pass@k, agreement captures the **diversity or determinism** of model outputs, revealing deeper failure modes and stability characteristics. While additional work is needed to generalize findings, this diagnostic complements existing evaluation tools and sheds light on **LLM behavior under budget constraints**.
