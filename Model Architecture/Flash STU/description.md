## Overview

This experiment compares a minimal **Spectral Transform Unit (STU)** layer with a simple **single-head self-attention** mechanism on a synthetic long-memory sequence prediction task. The STU uses **fixed 1D convolutional filters** derived from the **top eigenvectors of a Hankel matrix**, while the attention model uses standard dot-product self-attention.

## Task

Given a synthetic autoregressive sequence:

$x_{t+1} = 0.99 \cdot x_t + \sin(0.1 \cdot t) + \varepsilon_t,\quad \varepsilon_t \sim \mathcal{N}(0,\ 0.05^2)$


the goal is to predict the next value using a sliding window of length 50. A total of 500 sequences were generated, producing 125,000 input–target pairs.

## Models

- **STU Predictor:**
  - Convolves the input with 10 fixed Hankel-based filters.
  - Applies a linear projection at each time step.
  - Outputs the final time-step as the prediction.

- **Attention Predictor:**
  - Projects input to a learned feature space.
  - Applies one-head self-attention.
  - Uses the final time-step representation for prediction.

## Results

- **STU model** converged to an MSE of ~0.0073.
- **Attention model** plateaued at an MSE of ~8.4.
- STU consistently predicted values closer to the ground truth.

## Conclusion

The STU architecture dramatically outperformed the attention baseline, validating the Flash STU premise that **fixed spectral filters** can efficiently model long-range dependencies in time series with near-linear complexity.
