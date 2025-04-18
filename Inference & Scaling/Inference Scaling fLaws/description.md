## Overview

This study investigates the potential of confidence scores, derived from token-level probabilities, as a proxy for output quality in AI-generated code solutions. By analyzing 25 completions across five coding problems, the study explores how confidence-based filtering can mitigate overconfident errors (false positives) and enhance trust in AI-generated solutions. The findings suggest that confidence scores can offer a lightweight, interpretable complement to traditional resampling, contributing to both the utility and safety of LLM outputs.

## Task

- **Problem Selection:**
  - Five moderately complex coding problems were selected:
    - Check if a string is a palindrome
    - Calculate the factorial of a number
    - Generate Fibonacci sequence up to n terms
    - Check if a number is prime
    - Reverse a linked list
  
- **Sampling Responses:**
  - GPT-4-turbo was prompted five times per problem, resulting in 25 completions.
  - Each output was manually assessed for semantic correctness.

- **Confidence Extraction:**
  - Token-level log-probabilities were extracted and converted into probabilities to compute a mean confidence score for each response.

- **Error Classification:**
  - Responses were classified as:
    - **True Positives:** Confident and correct
    - **False Positives:** Confident but incorrect
    - **False Negatives:** Unconfident but correct
    - **True Negatives:** Unconfident and incorrect
  
- **Correlation Analysis:**
  - Confidence scores were compared across categories to evaluate their correlation with correctness, identifying patterns of overconfident errors and underconfident successes.

## Models & Setup

- **Model:** GPT-4-turbo
  - 25 total completions across five coding problems.
  - Confidence scores extracted from token-level probabilities.

- **Dataset:** 25 AI-generated responses for five coding problems.
  - Each problem was prompted five times.

## Metrics

- **Accuracy (Overall):** 72%
- **Average Confidence Score:** 0.8396
- **Error Classifications:**
  - False Positives (confident but wrong): 5
  - False Negatives (unconfident but right): 6
  - True Positives (confident and correct): 12
  - True Negatives (unconfident and wrong): 2

## Results

### 3.1 Accuracy and Confidence

| Metric             | Value  |
|--------------------|--------|
| Accuracy           | 72%    |
| Average Confidence | 0.8396 |

### 3.2 Overconfident Errors

| Problem                      | Confidence |
|------------------------------|------------|
| Reverse a linked list         | 0.9935     |
| Reverse a linked list         | 0.8240     |
| Fibonacci sequence            | 0.8168     |

### 3.3 Underconfident Correct Responses

| Problem                      | Confidence |
|------------------------------|------------|
| Factorial of a number         | 0.5582     |
| Factorial of a number         | 0.7015     |
| Prime number check            | 0.7781     |

## Conclusion

The results suggest that confidence scores correlate with correctness, supporting the hypothesis that confidence-based filtering can reduce false positives and improve output reliability. While high-confidence responses generally aligned with correctness, some highly confident responses were incorrect, highlighting the risk of overconfidence. Conversely, underconfident correct responses were often dismissed, suggesting an area for model improvement.

Confidence-based filtering offers a complementary approach to traditional resampling, providing a lightweight method to improve the safety and utility of LLMs. The findings also demonstrate the value of confidence estimation as an internal self-assessment tool, which could enhance the reliability of AI-generated code solutions in production contexts.

Future work could explore the integration of confidence scores with other strategies, such as semantic similarity or ensemble-based ranking, to further reduce errors and improve model trustworthiness in high-stakes applications.

