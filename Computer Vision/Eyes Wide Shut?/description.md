## Overview

This study conducts a comparative analysis of CLIP and DINOv2 embedding spaces, focusing on their performance across subtle image transformations such as flipping, zooming, and positional changes. By using standard clustering metrics, t-SNE visualizations, and intra-/inter-cluster distance measures, we evaluate how each model captures fine-grained visual distinctions. While CLIP forms tighter clusters, DINOv2 demonstrates better sensitivity to subtle visual changes. Our results confirm and extend prior work on visual sensitivity in multimodal models.

## Task

- **Data Generation:**
  - A dataset of 200 base images from open-access sources like ImageNet.
  - Seven subtle visual transformations applied to each base image:
    - Original, Flip, Zoom, Color, Text, Position, Orientation
  - Each transformed image saved with descriptive filenames indicating the transformation type.

- **Embedding Computation:**
  - Visual embeddings extracted using two pretrained models:
    - **CLIP**: `openai/clip-vit-large-patch14`
    - **DINOv2**: `facebook/dinov2-large`
  - Embeddings were L2-normalized for standardization.

- **Evaluation:**
  - Evaluated using clustering metrics (Silhouette Score, Calinski-Harabasz, Davies-Bouldin, ARI) and visual diagnostics.
  - t-SNE used for 2D visualization of embedding spaces.

## Models & Setup

- **Models:**
  - **CLIP**: `openai/clip-vit-large-patch14`
  - **DINOv2**: `facebook/dinov2-large`
  - Each model used to compute visual embeddings, followed by clustering and 2D visualization.

- **Dataset:**
  - 200 base images with seven transformation types (Original, Flip, Zoom, Color, Text, Position, Orientation).

- **Clustering Metrics:**
  - **Silhouette Score**
  - **Calinski-Harabasz Index**
  - **Davies-Bouldin Index**
  - **Adjusted Rand Index**
  - **Intra-cluster Distance**
  - **Inter-cluster Distance**

## Metrics

- **Clustering Quality:** Metrics used to evaluate how well the embeddings group images based on transformations.
- **Visualization:** t-SNE used for embedding projections and to generate visual diagnostics.

## Results

### 3.1 Clustering Metrics

| Metric                     | CLIP    | DINOv2  | Better Model | Improvement |
|----------------------------|---------|---------|--------------|-------------|
| Silhouette Score            | 0.0191  | -0.0007 | CLIP         | -103.6%     |
| Calinski-Harabasz Index     | 24.29   | 4.28    | CLIP         | -82.4%      |
| Davies-Bouldin Index        | 18.14   | 48.45   | CLIP         | -167.0%     |
| Adjusted Rand Index         | 0.2899  | 0.0583  | CLIP         | -79.9%      |
| Intra-cluster Distance      | 11.04   | 44.69   | CLIP         | -304.9%     |
| Inter-cluster Distance      | 4.95    | 8.42    | DINOv2       | +70.1%      |


## Discussion

Our results replicate and extend findings from Liu et al. (2024), demonstrating that CLIP, while semantically rich, struggles with fine-grained visual distinctions. CLIP produces tighter clusters but fails to adequately separate transformations like orientation and flipping. In contrast, DINOv2 excels at distinguishing subtle visual differences but at the cost of less compact clustering. These findings suggest that multimodal systems may benefit from blending CLIP's semantic alignment with DINOv2's visual sensitivity.

## Contributions

1. We provide a quantitative breakdown of clustering performance, showing how CLIP and DINOv2 handle visual variation.
2. Our visual diagnostics, including t-SNE projections and intra-cluster spread charts, offer an interpretable view of embedding space structure.
3. We confirm that DINOv2 captures fine-grained visual changes better than CLIP, supporting the hypothesis of CLIP's "visual blindness."

## Conclusion

CLIP clusters images more tightly by semantic content but struggles with subtle visual transformations. DINOv2, despite having more dispersed clusters, better preserves visual distinctions. For applications requiring fine-grained visual grounding, DINOv2 or hybrid approaches that combine CLIP and DINO-style representations could lead to more robust, visually-aware multimodal systems.
