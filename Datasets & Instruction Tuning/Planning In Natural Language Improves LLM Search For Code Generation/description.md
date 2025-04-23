## Overview

This study extends the PLANSEARCH framework to investigate the effects of sketch-guided planning in code generation using **GPT-4**. Using a subset of LiveCodeBench, we compare baseline prompting against natural language sketches that explicitly guide code structure and logic. The project explores whether lightweight sketching can improve both correctness and diversity under low-sample constraints.

## Task

- **Data Generation:**
  - Selected 3 challenging problems from LiveCodeBench.
  - For each problem, created a high-level natural language sketch to guide the code generation process.
  - Generated 3 completions per prompt for both baseline and sketch conditions using GPT-4.

- **Evaluation:**
  - Code outputs evaluated using pass@1 accuracy.
  - Behavioral diversity assessed via embedding-based similarity.
  - Qualitative review of structure and reasoning patterns across outputs.

## Models & Setup

- **Model:** GPT-4 via OpenAI API.
  - 3 completions per prompt, temperature set to 0.7 for diversity.
  
- **Dataset:** 3 LiveCodeBench problems with accompanying sketches.
  - Focused on reasoning-heavy tasks (e.g., dynamic programming, nested conditions).

## Metrics

- **Accuracy (pass@1):** Whether each generated code passes all tests for a given task.
- **Embedding Similarity:** SentenceTransformer-based similarity of code completions.
- **Qualitative Diversity:** Structural differences and reasoning variations across completions.

## Results

### 3.1 Accuracy (pass@1)

| Condition  | Accuracy |
|------------|----------|
| Baseline   | 0.33     |
| Sketch     | 0.67     |

### 3.2 Average Embedding Similarity

| Condition  | Similarity |
|------------|------------|
| Baseline   | 0.33       |
| Sketch     | 0.22       |

### 3.3 Observations

- Sketch-based generation produced more varied and semantically distinct outputs.
- Sketches improved correctness on two of the three problems.
- Qualitative review showed stronger reasoning alignment in sketch-guided completions.

## Conclusion

Natural language sketches offer a lightweight yet effective planning strategy for LLM-based code generation. Even with a small number of completions, sketch-guided prompts improved both accuracy and diversity. This supports PLANSEARCH’s core claim that interpretable guidance at inference time can meaningfully shape generation trajectories. Future work can expand the sketch vocabulary and scale evaluation to broader benchmarks.