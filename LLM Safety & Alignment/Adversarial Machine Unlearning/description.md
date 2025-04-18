## Overview

This study explores the effectiveness of machine unlearning methods, specifically RMU (as used in ZephyrRMU), in disrupting hazardous lexical clusters in large language models (LLMs). We evaluate the impact of RMU on the embedding space of hazardous tokens such as "bioweapon" and "exploit" in GPT-2 and ZephyrRMU. Results suggest that RMU reduces, but does not fully eliminate, dangerous capabilities by compressing semantic neighborhoods rather than erasing them.

## Task

- **Data Generation:**
  - Hazardous tokens: bioweapon, exploit, ransomware, etc.
  - Tokens encoded and analyzed using cosine similarity in the embedding space.
  - Nearest neighbors retrieved for each hazardous token.

- **Probing Method:**
  - Nearest-neighbor probe using cosine similarity in the embedding layer.
  - Models: GPT-2 Small (117M parameters) and ZephyrRMU (unlearning checkpoint).
  - 8 hazardous tokens analyzed, cosine similarity between hazardous probes and random tokens computed.

## Models & Setup

- **Model:** 
  - GPT-2 Small (117M parameters), unaligned.
  - ZephyrRMU (unlearning checkpoint from Henderson et al., 2025).
  
- **Methodology:**
  - Compute cosine similarity for 8 hazardous tokens and their top-15 nearest neighbors.
  - Filtered tokens: alphabetic, at least 3 characters, GPT-2 word-start marker.

## Results

### 3.1 Cosine Similarity with Hazardous Tokens

| Model      | Cosine Similarity for Hazardous Tokens |
|------------|----------------------------------------|
| GPT-2      | ~0.35                                  |
| ZephyrRMU  | ~0.1–0.2                               |

- **Observation:** ZephyrRMU shows partial disruption, but semantic clusters for hazardous tokens remain intact.

### 3.2 Top-15 Nearest Neighbors for Hazardous Tokens (GPT-2)

| Probe     | Top-15 Neighbors (examples)                   |
|-----------|----------------------------------------------|
| Bioweapon | externalToEVA, biochemical, biotech, bio-   |
| Anthrax   | anth, pathogens, ransomware, plutonium       |
| Malware   | ransomware, cybersecurity, viruses, hacking |
| Exploit   | exploit, vulnerabilities, manipulation       |

## Discussion

- **Key Insight:** RMU reduces the similarity between hazardous probes and their neighbors but does not erase the semantic structure entirely. This suggests that adversarial retrieval using synonym prompts may still succeed.

- **Figure Analysis:** The cosine similarity distributions in GPT-2 show a clear peak around 0.35 for hazardous tokens, while ZephyrRMU compresses this peak, indicating partial disruption.

## Conclusion

Machine unlearning methods like RMU reduce, but do not eliminate, the embedding-level alignment of hazardous lexical clusters. Future methods should focus on deeper embedding-space disruption to fully mitigate unsafe capabilities in LLMs.
