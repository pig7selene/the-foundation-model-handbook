# Technical Chapter Roadmap

Version 1.0 is published both as independent technical chapters and as the assembled handbook in `main.typ`. Chapter numbers follow the conceptual reading order and run strictly from 1 through 39.

## Core six-area reading map

1. **LLM Architecture:** Chapters 1--4 establish the token interface and decoder-only Transformer. Chapters 30--31 extend this foundation with attention execution, KV-representation design, and long-context connectivity.
2. **Pretraining:** Chapters 5--12 develop the objective, data, optimization, numerical stability, scaling, distributed execution, diagnostics, and practical FSDP state.
3. **Post-training / Alignment:** Chapters 13 and 15--22 develop SFT, preference learning, online alignment, evaluation, and distributed RL execution.
4. **Parameter-Efficient Fine-Tuning:** Chapter 14 follows SFT and compares frozen-base adaptation methods under the same data and masking contracts.
5. **Inference Optimization:** Chapters 23--31 move from autoregressive execution and KV state to quantization, scheduling, speculation, distributed inference, and efficient/long-context attention.
6. **LLM Systems:** Chapters 10--12, 22, and 28--29 form the cross-cutting systems path for training, FSDP state, distributed RL coordination, and serving.

**Retrieval-Augmented Generation:** Chapters 32--39 form the completed knowledge-augmentation extension and follow naturally after the inference and long-context foundations.

## Published chapters

### Part I — Foundations

1. **Tokenization and Input Representations.** Vocabulary design, reserved symbols, BPE, embedding lookup, positional inputs, token budgets, and implementation contracts.

### Part II — Architecture

2. **Transformer Architecture.** Decoder-only computation, residual streams, normalization placement, LM heads, shape contracts, and high-level resource costs.
3. **Attention and Position Encoding.** Scaled Dot-Product Attention, causal masking, MHA/MQA/GQA, RoPE, tensor shapes, and KV-cache accounting.
4. **Feed-Forward Networks, Normalization, and Residual Connections.** Position-wise and gated FFNs, activation functions, LayerNorm, RMSNorm, residual organization, initialization, gradient flow, and resource accounting.

### Part III — Pretraining

5. **Pretraining Objective and Language Modeling.** Autoregressive factorization, next-token prediction, logits, cross-entropy, teacher forcing, loss aggregation, perplexity, and implementation contracts.
6. **Pretraining Data.** Data sources, extraction, filtering, deduplication, mixtures, tokenization, packing, token budgets, governance, and implementation contracts.
7. **Optimization for Pretraining.** Mini-batch gradients, SGD, momentum, AdamW, learning-rate control, batch size, accumulation, clipping, optimizer-state memory, and implementation contracts.
8. **Numerical Precision and Training Stability.** Floating-point formats, mixed precision, stable reductions and cross-entropy, loss scaling, diagnostics, and numerical implementation contracts.
9. **Scaling Laws and Compute.** Parameter, token, and FLOP accounting; empirical power laws; compute-optimal allocation; scaling frontiers; planning limits; and reproducibility contracts.
10. **Distributed Training.** DDP, Tensor, Pipeline, and Sequence Parallelism; ZeRO and FSDP sharding; collective communication; memory accounting; scaling efficiency; and distributed execution contracts.
11. **Evaluation, Checkpointing, and Training Diagnostics.** Held-out validation and perplexity; online health metrics; failure localization; resumable and sharded checkpoints; recovery semantics; and experiment tracking.
12. **Practical FSDP and Distributed Training State.** FSDP execution lifecycle; parameter, gradient, and optimizer-state sharding; All-Gather and Reduce-Scatter; wrapping, mixed precision, activation checkpointing, offload, prefetch, checkpointing, restore, resharding, and implementation contracts.

### Part IV — Post-training, Alignment, and Adaptation

13. **Supervised Fine-Tuning.** Instruction-response demonstrations, Chat Templates, assistant-only loss masking, multi-turn supervision, data mixtures, full-parameter adaptation, LoRA, failure modes, evaluation, and implementation contracts.
14. **Parameter-Efficient Fine-Tuning.** LoRA and QLoRA; Adapters, Prefix Tuning, and Prompt Tuning; trainable-state and memory accounting; merging and multi-adapter serving; quantized-base adaptation; and implementation contracts.
15. **Preference Data and Reward Modeling.** Pairwise comparisons, preference collection, Bradley-Terry ranking, sequence-level Reward Models, calibration, evaluation, distribution shift, reward hacking, and implementation contracts.
16. **RLHF and PPO.** Autoregressive policies, on-policy Rollouts, policy gradients, actor-critic estimation, GAE, PPO clipping, reference-model KL regularization, reward shaping, stability, and implementation contracts.
17. **Direct Preference Optimization.** Reference-relative sequence likelihoods, the implicit-reward derivation, the DPO objective, offline preference learning, the DPO--PPO trade-off, data limitations, and implementation contracts.
18. **Group Relative Policy Optimization.** Grouped online Rollouts, relative rewards, critic-free advantages, PPO-style clipping and KL control, verifiable reward interfaces, failure modes, and implementation contracts.
19. **Reasoning RL, Rollouts, and Verifiable Rewards.** Sequential reasoning Rollouts, Outcome and Process Reward, deterministic verification, Best-of-$N$, Pass\@k, self-consistency, training- versus inference-time compute, failure modes, curriculum, and implementation contracts.
20. **Post-Training Evaluation and Alignment Trade-offs.** Multi-objective evaluation, human and model-based judging, verifiers, regression suites, proxy overoptimization, contamination, capability regression, alignment tax, and evaluation contracts.
21. **On-Policy Alignment and Iterative Policy Improvement.** Behavior-policy versions and rollout provenance; exploration, filtering, verifier-guided selection, Expert Iteration, DAPO, online failure modes, monitoring, and implementation contracts.
22. **Distributed RL Training Systems.** Actor, Rollout, Reward, Verifier, Critic, and Reference workers; training--inference separation; versioned trajectory data, placement, backpressure, queues, recovery, Ray, veRL, DataProto, FSDP, and inference-engine integration.

### Part V — Inference and Serving

23. **LLM Inference Fundamentals.** Autoregressive generation, Prefill and Decode, KV Cache accounting, sampling, context and batching, latency and throughput metrics, and inference contracts.
24. **KV Cache and Memory Optimization.** KV memory accounting, MHA/MQA/GQA cache layouts, paging and fragmentation, prefix reuse, cache lifecycle and eviction, KV Cache quantization, and implementation contracts.
25. **Quantization for LLM Inference.** Affine quantization, granularity and calibration, weight-only and activation-aware regimes, PTQ and QAT, outliers, LLM.int8(), SmoothQuant, GPTQ, AWQ, kernel trade-offs, and implementation contracts.
26. **Batching, Scheduling, and LLM Serving Systems.** Static, dynamic, and continuous batching; Prefill and Decode scheduling; queueing, admission, cache-aware capacity, preemption, fairness, and service metrics.
27. **Speculative Decoding and Inference Acceleration.** Draft-and-verify generation, exact speculative sampling, acceptance and residual correction, speed trade-offs, self-speculation, multi-token prediction, cache and serving interactions, and implementation contracts.
28. **Distributed LLM Inference and Parallelism.** Inference execution groups and replicas; Tensor, Pipeline, Sequence, and Expert Parallelism; serving collectives; Prefill--Decode disaggregation; topology, cache placement, scaling limits, and implementation contracts.
29. **Inference System Design and Performance Optimization.** Workload and SLO characterization; bottleneck classification; Roofline intuition; cache, quantization, batching, speculation, and distributed trade-offs; profiling, capacity planning, cost per token, and regression testing.

### Part VI — Efficient Attention and Long Context

30. **Efficient Attention and Head-Representation Design.** Attention IO and intermediate-state costs; tiled exact attention, online Softmax, FlashAttention, MHA/MQA/GQA/MLA, positional compatibility, KV Cache and Decode trade-offs, and implementation contracts.
31. **Sparse and Long-Context Attention.** Structured attention connectivity; local, sliding-window, global, block-sparse, strided, dilated, and random patterns; long-context constraints; RoPE extension; retrieval versus long context; and implementation contracts.

### Part VII — Retrieval-Augmented Generation and Knowledge Augmentation

32. **Retrieval-Augmented Generation Fundamentals.** External versus parametric knowledge; indexing and retrieval; corpus units, top-$k$ ranking, context construction, grounding, freshness, provenance, RAG--fine-tuning trade-offs, and implementation contracts.
33. **Embeddings and Semantic Retrieval.** Dense query and document embeddings; Dual-Encoders; similarity geometry and normalization; contrastive learning and negatives; semantic-retrieval storage, domain effects, failure modes, and implementation contracts.
34. **Vector Search and Approximate Nearest Neighbors.** Exact search and ANN recall; IVF, Product Quantization, and HNSW; candidate generation, filters, index lifecycle, memory and hardware trade-offs, and implementation contracts.
35. **Chunking and Document Segmentation.** Retrieval units and boundaries; fixed-length, linguistic, structure-aware, and semantic segmentation; chunk size and overlap; provenance, hierarchy, parent--child retrieval, and re-indexing.
36. **Sparse Retrieval and Hybrid Search.** Bag-of-Words, TF-IDF, BM25, and inverted indexes; sparse--dense complementarity; candidate union, score and rank fusion, RRF, learned sparse retrieval, filtering, and diagnosis.
37. **Reranking and Retrieval Refinement.** Candidate generation; Bi-Encoders and Cross-Encoders; ranking objectives; LLM reranking; diversity-aware context selection, cascades, latency, and failure diagnosis.
38. **Advanced Retrieval and RAG Architectures.** Query rewriting, expansion, Multi-Query Retrieval, decomposition, HyDE, feedback-driven and multi-hop retrieval; hierarchical and graph-based retrieval; adaptive routing, evidence aggregation, conflict handling, and implementation contracts.
39. **RAG Evaluation and Diagnostics.** Component and end-to-end evaluation; retrieval, context, generation, grounding, and citation metrics; ablations, error attribution, evaluation data, regression suites, online signals, cost, and implementation contracts.

## Future editions

Future material will be scoped as a later edition rather than silently folded into the v1.0 core roadmap. A new chapter must declare prerequisites and sources, remain independently readable, and be incorporated into the assembled edition only through an explicit manifest update.
