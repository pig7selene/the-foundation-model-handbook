# Technical Chapter Roadmap

This repository publishes independent technical chapters rather than a single assembled book. The roadmap records a coherent learning sequence, but it does not create source directories until a chapter is ready to be researched, written, and visually reviewed.

## Published

### Foundations

1. **Tokenization and Input Representations.** Vocabulary design, reserved symbols, BPE, embedding lookup, positional inputs, token budgets, and implementation contracts.

### Architecture

2. **Transformer Architecture.** Decoder-only computation, residual streams, normalization placement, LM heads, shape contracts, and high-level resource costs.
3. **Attention and Position Encoding.** Scaled Dot-Product Attention, causal masking, MHA/MQA/GQA, RoPE, tensor shapes, and KV-cache accounting.
4. **Feed-Forward Networks, Normalization, and Residual Connections.** Position-wise and gated FFNs, activation functions, LayerNorm, RMSNorm, residual organization, initialization, gradient flow, and resource accounting.

### Pretraining

5. **Pretraining Objective and Language Modeling.** Autoregressive factorization, next-token prediction, logits, cross-entropy, teacher forcing, loss aggregation, perplexity, and implementation contracts.
6. **Pretraining Data.** Data sources, extraction, filtering, deduplication, mixtures, tokenization, packing, token budgets, governance, and implementation contracts.
7. **Optimization for Pretraining.** Mini-batch gradients, SGD, momentum, AdamW, learning-rate control, batch size, accumulation, clipping, optimizer-state memory, and implementation contracts.
8. **Numerical Precision and Training Stability.** Floating-point formats, mixed precision, stable reductions and cross-entropy, loss scaling, diagnostics, and numerical implementation contracts.
9. **Scaling Laws and Compute.** Parameter, token, and FLOP accounting; empirical power laws; compute-optimal allocation; Kaplan- and Chinchilla-style frontiers; planning limits; and reproducibility contracts.
10. **Distributed Training.** DDP, Tensor, Pipeline, and Sequence Parallelism; ZeRO and FSDP sharding; collective communication; memory accounting; scaling efficiency; and distributed execution contracts.
11. **Evaluation, Checkpointing, and Training Diagnostics.** Held-out validation and perplexity; online health metrics; failure localization; fully resumable and sharded checkpoints; recovery semantics; checkpoint selection; and reproducible experiment tracking.

### Post-training

12. **Supervised Fine-Tuning.** Instruction-response demonstrations, Chat Templates, assistant-only loss masking, multi-turn supervision, data mixtures, full-parameter adaptation, LoRA, failure modes, evaluation, and implementation contracts.
13. **Preference Data and Reward Modeling.** Pairwise comparisons, preference collection, Bradley-Terry ranking, sequence-level Reward Models, calibration, evaluation, distribution shift, reward hacking, and implementation contracts.
14. **RLHF and PPO.** Autoregressive policies, on-policy Rollouts, policy gradients, actor-critic estimation, GAE, PPO clipping, reference-model KL regularization, reward shaping, stability, and implementation contracts.
15. **Direct Preference Optimization.** Reference-relative sequence likelihoods, the implicit-reward derivation, the DPO objective, offline preference learning, the DPO--PPO trade-off, data limitations, and implementation contracts.
16. **Group Relative Policy Optimization.** Grouped online Rollouts, relative rewards, critic-free group-normalized advantages, PPO-style clipping and KL control, verifiable reward interfaces, failure modes, and implementation contracts.
17. **Reasoning RL, Rollouts, and Verifiable Rewards.** Sequential reasoning Rollouts, Outcome and Process Reward, deterministic verification, Best-of-$N$, Pass\@k, self-consistency, training- versus inference-time compute, failure modes, curriculum, and implementation contracts.
18. **Post-Training Evaluation and Alignment Trade-offs.** Multi-objective evaluation, human and model-based judging, verifiers, regression suites, proxy overoptimization, contamination, capability regression, alignment tax, and evaluation contracts.

### Inference and Serving

19. **LLM Inference Fundamentals.** Autoregressive generation, Prefill and Decode, KV Cache accounting, sampling policies, context and batching, latency and throughput metrics, and inference implementation contracts.
20. **KV Cache and Memory Optimization.** KV memory accounting, MHA/MQA/GQA cache layouts, paging and fragmentation, prefix reuse, cache lifecycle and eviction, KV Cache quantization, and implementation contracts.
21. **Quantization for LLM Inference.** Affine quantization, granularity and calibration, weight-only and activation-aware regimes, PTQ and QAT, outliers, LLM.int8(), SmoothQuant, GPTQ, AWQ, kernel trade-offs, and implementation contracts.
22. **Batching, Scheduling, and LLM Serving Systems.** Static, dynamic, and continuous batching; Prefill and Decode scheduling; queueing, admission, cache-aware capacity, preemption, fairness, service metrics, and implementation contracts.
23. **Speculative Decoding and Inference Acceleration.** Draft-and-verify generation, exact speculative sampling, acceptance and residual correction, speed trade-offs, self-speculation, multi-token prediction, cache and serving interactions, and implementation contracts.
24. **Distributed LLM Inference and Parallelism.** Inference execution groups and replicas; Tensor, Pipeline, Sequence, and Expert Parallelism; serving collectives; Prefill--Decode disaggregation; topology, cache placement, scaling limits, and implementation contracts.

## Planned directions

- **Inference and Serving:** attention kernels and latency-throughput optimization.
- **Retrieval-Augmented Generation:** retrieval pipelines, embedding indexes, reranking, context construction, grounding, and retrieval evaluation.
- **Agents and Tool Use:** tool interfaces, planning, execution loops, memory, environment interaction, and agent evaluation.
- **Multimodal Models:** vision-language inputs, multimodal tokenization and fusion, training objectives, evaluation, and system interfaces.

Each future chapter should declare its scope, prerequisites, and sources in its own directory and should remain readable without a combined-book build.
