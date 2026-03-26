# Tiny Aya Safety Representations

Interpretability research on safety mechanisms in [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) (Tiny Aya Global, 3.35B), focusing on how harmfulness and refusal are encoded across languages.

## Notebooks

- [notebooks/zhao_replication_tinyaya.ipynb](notebooks/zhao_replication_tinyaya.ipynb) — Replication of Zhao et al. (2025) *"LLMs Encode Harmfulness and Refusal Separately"* on Tiny Aya. Finds harmfulness and refusal directions are nearly orthogonal (~0.095 cosine sim), but safety is heavily template-anchored (refusal drops from 85% → 11% without chat template).
- [notebooks/sae_tiny_aya_starter.ipynb](notebooks/sae_tiny_aya_starter.ipynb) — Sparse Autoencoder (TopK, k=64, 16K dictionary) trained on middle-layer residual stream activations. Identifies cross-lingual refusal features that activate on unsafe prompts across 5 languages.

## Docs

- [docs/Decoupling Harmfulness from Refusal in Tiny Aya — Results.md](<docs/Decoupling Harmfulness from Refusal in Tiny Aya — Results.md>) — Full results writeup for the Zhao et al. replication.
- [docs/inital_SAE_results.md](docs/inital_SAE_results.md) — Preliminary SAE analysis results.

## Author

Abrit Pal Singh — March 2026
