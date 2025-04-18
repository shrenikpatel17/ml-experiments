## Overview

This study adapts the **INSTRUCT-SKILLMIX pipeline** to generate synthetic safety and alignment instruction-response pairs using **GPT-4o**, and fine-tunes **GPT-2 base (124M parameters)** on this data. The fine-tuned model is compared against the pre-trained GPT-2 baseline using identical prompts. With just 280 high-quality examples, the experiment explores improvements in safety and alignment responses. Results suggest noticeable gains, with implications for low-resource fine-tuning.

## Task

- **Data Generation:**
  - 280 instruction-response pairs focused on safety and alignment, generated using GPT-4o.
  - Each response was limited to 750 words.
  - The dataset was curated using the synthetic data generation approach from INSTRUCT-SKILLMIX.
  - The 35 alignment and safety skills were prioritized in the dataset.

- **Fine-tuning:**
  - Fine-tuning GPT-2 base (124M parameters) on the generated dataset.
  - The model was fine-tuned with specific hyperparameters to encourage diverse and contextually relevant responses.
  
- **Evaluation:**
  - Evaluated using qualitative and quantitative measures, including average scores, LCWR (Length-Controlled Win Rate), and repetition metrics.

## Models & Setup

- **Model:** GPT-2 base (124M parameters)
  - Fine-tuning applied to 280 instruction-response pairs.
  - Hyperparameters: 3 training epochs, batch size of 4, dynamic save strategies.
  
- **Dataset:** 280 synthetic instruction-response pairs focused on safety and alignment.
  - Prioritized alignment and safety skills.

## Metrics

- **Qualitative Evaluation:** Responses rated on a scale of 1-5 for alignment with instructions.
- **LCWR (Length-Controlled Win Rate):** Compared the effectiveness of the fine-tuned GPT-2 and the GPT-4o model.
- **Repetition Metric:** Measured sentence similarity to evaluate redundancy in responses.

## Results

### 3.1 Qualitative Measurements

| Model            | Average Score |
|------------------|---------------|
| GPT-2 Base       | 2.5           |
| Fine-tuned Model | 4.6           |

### 3.2 Quantitative Measurements

| Model Comparison                | Fine-tuned Model | GPT-2 Base |
|----------------------------------|------------------|------------|
| Fine-tuned Model vs GPT-2 Base   | 100%             | 0%         |

### 3.3 Average Repetition Scores

| Model            | Average Repetition Score |
|------------------|--------------------------|
| GPT-2 Base       | 0.294                    |
| Fine-tuned Model | 0.180                    |

## Conclusion

The fine-tuned model outperforms the baseline, demonstrating that small, targeted datasets can significantly enhance safety and alignment. This aligns with findings from INSTRUCT-SKILLMIX, where instruction-following was improved with synthetic data. The fine-tuned model not only scored higher in qualitative evaluations but also showed better performance in LCWR and reduced repetition in generated responses. Future work could scale the dataset size and fine-tuning process to further improve model performance.
