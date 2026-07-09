DSP-LM: Deterministic Structural Plasticity Language Model

& The "Quality of Thought" Streaming Curriculum

This repository contains the training architecture for the DSP-LM, a custom PyTorch language model that entirely replaces standard Multi-Head Attention with biological, topology-based feature branches and causal resonators.

Coupled with a built-in Quality of Thought Curriculum, this project forces the network's internal geometry to lock into strict, formal causal rules before it is allowed to process abstract semantic noise.

🧠 Architectural Paradigm: DSP-LM

The DSP-LM fundamentally redesigns the standard Transformer block, moving away from the "point-neuron" abstraction and $O(N^2)$ attention bottlenecks. It synthesizes concepts from neurobiology, optical metrology (3D QPI), and memristive hardware (computing-in-memory) into a fully vectorized, GPU-accelerated PyTorch model.

Core Mechanisms

Dendritic Branching Logic: Instead of a single weighted sum, each artificial neuron is composed of multiple independent "branches." Each branch possesses its own hidden dimension, allowing it to locally solve complex non-linear logic (like XOR operations) before integrating.

Causal Resonant Field Mixing (Diagonal SSM): Token sequences are treated as a driven 1D latent field using damped resonators — a diagonal, complex-pole State-Space Model (S4D-style). Each channel is a stable damped oscillator (pole = -exp(a) + i·ω: negative real part damps, imaginary part sets the resonant frequency). The response is applied as a causal FFT convolution, giving an effectively unbounded receptive field at O(N log N) cost with none of the quadratic memory blowup of self-attention. This replaces the earlier dilated-convolution prototype, whose fixed kernel could only see ~100 tokens regardless of input length — which is why long sequence lengths now train correctly.

Asymmetric Error Cost Gating (The Soma): Each branch produces a value _vector_; a steep, differentiable sigmoid gate decides whether that branch has structurally resolved and, if not, suppresses its whole vector toward zero (Type II Error avoidance — dropping unresolved/hallucinated states). The soma then integrates the surviving branch vectors back to the model dimension. (Earlier prototypes collapsed each branch to a single scalar before integrating, wasting almost all of the branch capacity; the gate now acts on the full vector.)

Vectorized Topology: To achieve extreme parameter density without the latency of Python loops, the entire physical geometry of the branches is collapsed into multi-dimensional tensors processed simultaneously via Einstein Summation (torch.einsum).

📚 The "Quality of Thought" Streaming Curriculum

Instead of training on a randomized, entangled corpus, the DSP-LM script utilizes datasets.interleave_datasets in streaming mode to dynamically mix HuggingFace data without crashing system RAM. Documents are token-packed into full-length blocks (no padding waste), and each dataset uses a schema-aware extractor so the model trains on real content rather than titles or task labels.

Dataset slate (all non-gated, all stream cleanly, balanced "curriculum of courses"):

| Phase      | Dataset                                      | Size    | Style                                                                                                                                          |
| ---------- | -------------------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Logic      | `reasoning-core/procedural-pretraining-pile` | ~7.3 GB | Procedural formal reasoning, correct-by-design (PDDL, FOL, CFG, causal). Its generator scales to trillions of tokens; the hosted file is 7 GB. |
| Math       | `HuggingFaceTB/cosmopedia` · `khanacademy`   | ~108 MB | Khan Academy course prose (synthetic, Mixtral-generated).                                                                                      |
| Physics    | `HuggingFaceTB/cosmopedia` · `openstax`      | ~668 MB | OpenStax textbook prose, including University Physics.                                                                                         |
| Philosophy | `sayhan/strix-philosophy-qa`                 | ~391 MB | 134K philosophy Q&A pairs derived from the SEP.                                                                                                |

Because the pipeline **streams** and packs, total pool size barely matters — you only ever pull what `steps × batch × seq_len` demands (tens of MB per substep), so there is no multi-GB download. The earlier slate mixed a 56 GB math corpus with a 30 MB gated physics set; the small one would have been cycled and memorised while the gated one failed to load at all. Scale-up swaps (still non-gated) are noted in the `Config.repos` comments: math → `cosmopedia:auto_math_text` (~8.8 GB), humanities → `cosmopedia:stanford` (~6.3 GB).

The curriculum routes the model through four distinct phases:

Phase 1: Logic & Formal Reasoning (Dominant) * Mixture: 90% Logic (reasoning-core/procedural-pretraining-pile), 10% Math

Goal: Establishes rigid physical pathways for First-Order Logic (FOL) and structural causality.

Phase 2: Mathematical & Scientific Prose

Mixture: Interpolates from 30% Logic / 70% Math to a 50/50 split (Khan Academy course prose via Cosmopedia).

Goal: Adapts logical geometries to the syntax of formal proofs and symbolic notation.

Phase 3: Dedicated Physics Application

Mixture: Shifts to 15% Logic / 25% Math / 60% Physics (OpenStax textbook prose via Cosmopedia).

Goal: Applies causal mathematical modeling to physical world constraints.

Phase 4: Philosophy & Meta-Reflection

Mixture: Final 25/25/25/25 mix introducing Conceptual Analysis (sayhan/strix-philosophy-qa).

Goal: Routes highly abstract, semantic noise through the previously established rigorous logical bottlenecks.

🚀 Quick Start (Google Colab / Colab Pro)

The entire architecture and curriculum are condensed into a single script (`colab_trainable_dendritic_lm.py`). It is designed to be executed in a high-VRAM environment (an A100 on Colab Pro) utilizing PyTorch bfloat16 Automatic Mixed Precision plus gradient checkpointing so long (2048-token) sequences fit in memory. Convert it to a notebook with `jupytext --to notebook colab_trainable_dendritic_lm.py`.

```
jupytext --to notebook --output Colab_Trainable_Dendritic_LM.ipynb colab_trainable_dendritic_lm.py
```

1. Install Requirements

In your first Colab cell, install the necessary HuggingFace libraries:

pip install torch transformers datasets accelerate

2. Run the Training Loop

Run the script directly. It will automatically initialize the GPT-2 Tokenizer, mount the streaming datasets, and begin the multi-phase curriculum.

python colab_trainable_dendritic_lm.py

3. Checkpoints & Outputs

Checkpoints (the model's state_dict and the tokenizer) are automatically saved to ./checkpoints/DSP_LM/ at the end of every curriculum substep.

Tip for Colab Users: Mount your Google Drive and change base_output_dir in the script to save weights directly to your cloud storage.

🔬 References & Theoretical Foundations

This architecture synthesizes breakthroughs from the following papers and concepts:

Dendritic Computational Independence: Research from the Hebrew University of Jerusalem detailing the "Functional Complexity Index (FCI)" of biological pyramidal neurons.

Phase-Change Memristor Arrays (Computing-In-Memory): Hardware topologies that natively bypass the von Neumann bottleneck.

3D Quantitative Phase Imaging (QPI): Label-free volumetric mapping of complex biological topologies (UCLA, Advanced Photonics 2024).

ResonatorLM: Causal Resonant Field Mixing to eliminate Self-Attention (alphaXiv:2607.05583).

Diagonal State-Space Models (S4D): Gu, Gupta, Goel, Ré, "On the Parameterization and Initialization of Diagonal State Space Models" (NeurIPS 2022) — the damped-complex-pole resonator formulation used for the temporal mixer.

Imbalanced Cognitive Curricula: Pretraining phases enabling selective, disentangled feature representation (alphaXiv:2607.04846).
