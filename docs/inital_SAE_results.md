# Preliminary Results: Sparse Autoencoder Analysis (v2)

## Cross-Lingual Refusal Features in Tiny Aya Global

**Author:** Abrit | **Date:** March 2026 | **Status:** _Exploratory / Proof-of-Concept_

---

> **TL;DR** — We trained a TopK Sparse Autoencoder (k=64, 16K dictionary) on Tiny Aya Global's middle-layer residual stream activations using unsafe prompts from the MultiJail benchmark (Deng et al., ICLR 2024) and balanced safe prompts across 5 languages. Multiple SAE features activate strongly on unsafe prompts across all languages while remaining silent on safe prompts — providing preliminary evidence for a predominantly language-agnostic refusal mechanism in Tiny Aya's safety architecture.

---

## 1. Motivation

Our MultiJail benchmark evaluation of the Tiny Aya family (Global, Fire, Water, Earth) shows remarkably consistent safe response rates of 88–91% across 10 languages, with standard deviations of only 3–5%. This is in stark contrast to models like SmolLM3-3b (~49% mean, 26% std) and Qwen3-4b (~86% mean, but collapsing to 1.6% on Swahili with 91% invalid responses).

The behavioral data tells us _that_ Tiny Aya refuses consistently across languages, but not _how_. Does the model use a single language-agnostic refusal mechanism, or independent per-language safety features? Sparse Autoencoders can answer this by decomposing internal activations into interpretable features.

## 2. Experimental Setup

|Parameter|Value|
|---|---|
|**Model**|CohereLabs/tiny-aya-global (instruction-tuned, 3.35B params, Cohere2 architecture)|
|**Hidden Size**|2048|
|**Target Layer**|Layer 18 of 36 (50% depth), residual stream, last token position|
|**SAE Architecture**|TopK SAE (k=64, dictionary size 16,384, 8x overcomplete)|
|**Unsafe Prompts**|10 per language from MultiJail (Deng et al., ICLR 2024), 50 total|
|**Safe Prompts**|10 per language (control set), 50 total|
|**Languages**|English (en), Thai (th), Korean (ko), Arabic (ar), Swahili (sw)|
|**Training**|200 epochs, Adam (lr=3e-4), batch size 32, reconstruction loss only|
|**Compute**|Google Colab free tier, Tesla T4 (16GB VRAM)|

**Key design choices:**

- **Instruction-tuned model, not base:** Refusal behavior only emerges after SFT + preference training. SAEs trained on base model activations would lack features for safety-relevant concepts introduced during alignment.
- **MultiJail over XSafety:** We chose MultiJail because it covers all 5 of our target languages. Only 2 (English, Arabic) overlap with XSafety's coverage, making MultiJail the better fit for cross-lingual validation.
- **Balanced classes:** Earlier experiments with a 5:1 unsafe-to-safe ratio produced features that conflated language identity with safety signal. Balancing to 1:1 resolved this.

## 3. Iteration History

This is the result of several iterative refinements:

|Run|Unsafe Source|Safe Source|Ratio|Key Issue|
|---|---|---|---|---|
|v1 (ReLU SAE)|Placeholder (self-written, 10/lang)|Placeholder (10/lang)|1:1|L0 ~2000, couldn't enforce sparsity with L1 penalty|
|v2 (TopK SAE)|Placeholder (10/lang)|Placeholder (10/lang)|1:1|Clean results, but prompts weren't from a published benchmark|
|v3 (TopK SAE)|MultiJail (50/lang)|Placeholder (10/lang)|5:1|Class imbalance caused language-identity features to leak into safety signal|
|**v4 (TopK SAE)**|**MultiJail (10/lang)**|**Placeholder (10/lang)**|**1:1**|**Current run — balanced, real benchmark data**|

The switch from ReLU to TopK was necessary because with only a few hundred data points, L1 sparsity tuning was ineffective — even at L1=5e-1, L0 remained at ~380. TopK directly enforces exactly 64 active features per input, eliminating this problem entirely.

## 4. Training Dynamics

The TopK architecture enforces exactly 64 active features per input. Final reconstruction loss converged to ~0.019 with dead features at 23.9% by epoch 200, indicating the SAE is making reasonable use of its dictionary despite the small dataset size (100 prompts against a 16K dictionary).

![](./assets/Pasted%20image%2020260317065209.png)
 _Figure 1: Training curves for the balanced MultiJail run. L0 locked at 64 by TopK. Dead features drop steadily from ~90% to 23.9% as the SAE discovers useful features._

## 5. Key Finding: Language-Agnostic Refusal Features

The central result is the cross-lingual activation heatmap below. The left panel shows mean feature activation on unsafe prompts across 5 languages; the right panel shows the same features on safe prompts.

![](./assets/Pasted%20image%2020260317065109.png)
 _Figure 2: Cross-lingual feature activation heatmap. Left: unsafe prompts (red = high activation). Right: safe prompts (same features, near-zero). Features like F10194, F7197, F5702, F12147, F15515, F14201 fire across all 5 languages on unsafe inputs only._

**Observations:**

- **Universal refusal features:** Features F10194, F7197, F5702, F7078, and F12147 all scored 100% on the cross-lingual consistency metric — they activate more on unsafe than safe prompts in every one of the five languages. The top feature (F10194) has an unsafe mean activation of 2.70 vs a safe mean of 0.01, with 86% activation frequency on unsafe prompts and just 2% on safe. This pattern suggests a shared, language-agnostic safety mechanism in the model's middle layers.
    
- **Uneven strength across languages:** Activation intensity varies — for F10194, English shows the strongest unsafe activation (3.40) while other languages range from 1.8 to 3.0. This mirrors the slight variation in behavioral refusal rates (93% en/th vs 87% ar in the MultiJail benchmark), suggesting the shared mechanism exists but is applied with uneven strength.
    
- **Language-entangled features:** F7078 fires strongly on unsafe prompts (especially Arabic and Swahili) but also activates on safe Thai and Swahili. This likely encodes linguistic patterns that partially correlate with the safety signal rather than being a pure refusal feature.
    
- **Clean separation improved with balancing:** The right panel is significantly cleaner than our earlier 5:1 imbalanced run, where language-identity features were leaking into the safety signal. Balanced classes make the contrastive signal much more reliable.
    

## 6. Validation Against Placeholder Prompts

An important validation: our initial experiment used self-written placeholder prompts, while this run uses real MultiJail benchmark data (manually translated by native speakers, covering 18 safety categories including violence, discrimination, hate speech, and more). The core finding — universal refusal features that fire across all 5 languages — persists across both runs. The specific feature IDs differ (expected, since SAE training is stochastic), but the structural pattern is consistent. This gives us confidence the signal is genuine rather than an artifact of prompt selection.

## 7. Feature-Level Safety Classification (AUC)

We computed the AUC of each SAE feature as a binary classifier of unsafe vs. safe prompts.

![](./assets/Pasted%20image%2020260317065232.png)
Figure 3: Distribution of per-feature AUC scores. The broad spread from 0.1 to 0.9 indicates many features carry genuine safety signal._

The top features achieve strong AUC scores: F5702 (0.93), F10194 (0.93), F7078 (0.93), F7197 (0.92), F12147 (0.91), F15515 (0.89), and F14201 (0.85). In total, 45 features have AUC > 0.7 or < 0.3, indicating strong safety discriminators in both directions. The AUC distribution shows a broad spread in both tails compared to the sharp 0.5 spike in earlier imbalanced runs, confirming that balanced classes allow safety-relevant features to emerge properly.

## 8. Limitations & Caveats

- **Small dataset:** 10 unsafe + 10 safe prompts per language (100 total). This is a proof-of-concept; the full 315 MultiJail prompts per language with matched safe prompts would provide much stronger statistical power.
    
- **Safe prompts not from a published benchmark:** The safe (control) prompts are self-written translated questions, not from a published dataset. Flores-200 or another multilingual benchmark would be a more rigorous control set.
    
- **Single layer, single variant:** We only analyzed layer 18 (of 36) of Tiny Aya Global. The safety mechanism may span multiple layers, and regional variants (Fire, Water, Earth) may differ.
    
- **No causal verification:** We have shown correlation (features activate on unsafe prompts) but not causation. Feature steering experiments (clamping refusal features to zero and checking if the model stops refusing) are needed.
    
- **Stochastic training:** Specific feature IDs are not stable across runs. The structural patterns (universal vs. language-specific) are consistent, but individual features should not be treated as fixed identifiers.
    

## 9. Next Steps

- **Scale to full MultiJail:** Use all 315 prompts per language with matched safe prompts from Flores-200 for full statistical power.
    
- **Multi-layer analysis:** Collect activations at layers 9, 18, 27 (25%, 50%, 75% depth) to trace where refusal features first emerge and how they evolve across layers.
    
- **Compare regional variants:** Run the same pipeline on Tiny Aya Fire, Water, and Earth — do they share the same refusal features or develop variant-specific ones? This is the most distinctive contribution for the Expedition submission.
    
- **Causal steering:** Clamp top refusal features to zero and verify the model stops refusing unsafe prompts. This converts correlation into causation.
    
- **Per-category analysis:** MultiJail's safety category tags (violence, discrimination, hate speech, etc.) enable testing whether different harm types activate different features — or whether a single mechanism handles all unsafe content.
    
- **Connect to representational analysis:** Use CKA and probing to validate whether SAE-identified refusal features correspond to cross-lingual representational convergence in the broader Expedition analysis being conducted by the team.
    

## 10. References

- Deng, Y., Zhang, W., Pan, S. J., & Bing, L. (2024). Multilingual Jailbreak Challenges in Large Language Models. _ICLR 2024_. Data: github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs
- Wang, W., Tu, Z., Chen, C., Yuan, Y., Huang, J., Jiao, W., & Lyu, M. R. (2024). All Languages Matter: On the Multilingual Safety of Large Language Models. _ACL Findings 2024_.
- Cohere Labs. (2026). Tiny Aya: Bridging Scale and Multilingual Depth. huggingface.co/CohereLabs/tiny-aya-global
- Rajamanoharan, S., et al. (2024). Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders. _arXiv:2407.14435_.
- Gao, L., et al. (2024). Scaling and Evaluating Sparse Autoencoders. _OpenAI Technical Report_.
