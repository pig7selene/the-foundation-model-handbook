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
38. **Practical FSDP and Distributed Training State.** FSDP execution lifecycle; parameter, gradient, and optimizer-state sharding; All-Gather and Reduce-Scatter; wrapping granularity, mixed precision, activation checkpointing, offload, and prefetch; peak-memory and communication diagnosis; full and sharded state dictionaries; distributed checkpointing, restore, resharding, and implementation contracts. This chapter applies Chapter 10's distributed-training model and Chapter 11's resumability requirements without changing their scope.

### Post-training

12. **Supervised Fine-Tuning.** Instruction-response demonstrations, Chat Templates, assistant-only loss masking, multi-turn supervision, data mixtures, full-parameter adaptation, LoRA, failure modes, evaluation, and implementation contracts.
13. **Preference Data and Reward Modeling.** Pairwise comparisons, preference collection, Bradley-Terry ranking, sequence-level Reward Models, calibration, evaluation, distribution shift, reward hacking, and implementation contracts.
14. **RLHF and PPO.** Autoregressive policies, on-policy Rollouts, policy gradients, actor-critic estimation, GAE, PPO clipping, reference-model KL regularization, reward shaping, stability, and implementation contracts.
15. **Direct Preference Optimization.** Reference-relative sequence likelihoods, the implicit-reward derivation, the DPO objective, offline preference learning, the DPO--PPO trade-off, data limitations, and implementation contracts.
16. **Group Relative Policy Optimization.** Grouped online Rollouts, relative rewards, critic-free group-normalized advantages, PPO-style clipping and KL control, verifiable reward interfaces, failure modes, and implementation contracts.
17. **Reasoning RL, Rollouts, and Verifiable Rewards.** Sequential reasoning Rollouts, Outcome and Process Reward, deterministic verification, Best-of-$N$, Pass\@k, self-consistency, training- versus inference-time compute, failure modes, curriculum, and implementation contracts.
18. **Post-Training Evaluation and Alignment Trade-offs.** Multi-objective evaluation, human and model-based judging, verifiers, regression suites, proxy overoptimization, contamination, capability regression, alignment tax, and evaluation contracts.
37. **On-Policy Alignment and Iterative Policy Improvement.** Behavior-policy versions and rollout provenance; exploration, filtering, verifier-guided selection, and Expert Iteration; DAPO dynamic sampling, asymmetric clipping, token-level loss, and overlong-response shaping; online failure modes, monitoring, and implementation contracts. This chapter extends the algorithmic Post-training sequence without covering the distributed rollout-worker architecture reserved for Chapter 39.
39. **Distributed RL Training Systems.** Actor, Rollout, Reward, Verifier, Critic, and Reference worker roles; training--inference separation; versioned trajectory data, policy synchronization, bounded rollout staleness, placement, backpressure, queues, and recovery; Ray resource orchestration; veRL and DataProto as concrete examples; FSDP and inference-engine integration; observability; and implementation contracts. This chapter applies the algorithms of Chapters 14, 16, 17, and 37 together with the distributed-state model of Chapter 38.

### Inference and Serving

19. **LLM Inference Fundamentals.** Autoregressive generation, Prefill and Decode, KV Cache accounting, sampling policies, context and batching, latency and throughput metrics, and inference implementation contracts.
20. **KV Cache and Memory Optimization.** KV memory accounting, MHA/MQA/GQA cache layouts, paging and fragmentation, prefix reuse, cache lifecycle and eviction, KV Cache quantization, and implementation contracts.
21. **Quantization for LLM Inference.** Affine quantization, granularity and calibration, weight-only and activation-aware regimes, PTQ and QAT, outliers, LLM.int8(), SmoothQuant, GPTQ, AWQ, kernel trade-offs, and implementation contracts.
22. **Batching, Scheduling, and LLM Serving Systems.** Static, dynamic, and continuous batching; Prefill and Decode scheduling; queueing, admission, cache-aware capacity, preemption, fairness, service metrics, and implementation contracts.
23. **Speculative Decoding and Inference Acceleration.** Draft-and-verify generation, exact speculative sampling, acceptance and residual correction, speed trade-offs, self-speculation, multi-token prediction, cache and serving interactions, and implementation contracts.
24. **Distributed LLM Inference and Parallelism.** Inference execution groups and replicas; Tensor, Pipeline, Sequence, and Expert Parallelism; serving collectives; Prefill--Decode disaggregation; topology, cache placement, scaling limits, and implementation contracts.
25. **Inference System Design and Performance Optimization.** Workload and SLO characterization; bottleneck classification; Roofline intuition; cache, quantization, batching, speculation, and distributed trade-offs; profiling, capacity planning, cost per token, regression testing, and implementation contracts.

### Retrieval-Augmented Generation (complete)

26. **Retrieval-Augmented Generation Fundamentals.** External versus parametric knowledge; offline indexing and online retrieval; corpus units, top-$k$ ranking, context construction, grounded generation, retrieval quality, freshness, provenance, RAG--fine-tuning trade-offs, and implementation contracts.
27. **Embeddings and Semantic Retrieval.** Dense query and document embeddings; Dual-Encoders; similarity geometry and normalization; contrastive learning and negatives; semantic-retrieval storage, domain effects, failure modes, and implementation contracts.
28. **Vector Search and Approximate Nearest Neighbors.** Exact search and ANN recall; IVF, Product Quantization, and HNSW; candidate generation, filters, index lifecycle, memory and hardware trade-offs, vector databases, and implementation contracts.
29. **Chunking and Document Segmentation.** Retrieval units and boundaries; fixed-length, linguistic, structure-aware, and semantic segmentation; chunk size and overlap; provenance, hierarchy, parent--child retrieval, special document forms, re-indexing, and implementation contracts.
30. **Sparse Retrieval and Hybrid Search.** Bag-of-Words, TF-IDF, BM25, and inverted indexes; sparse--dense complementarity; candidate union, score and rank fusion, RRF, learned sparse retrieval, filtering, failure diagnosis, and implementation contracts.
31. **Reranking and Retrieval Refinement.** Recall-oriented candidate generation; Bi-Encoders and Cross-Encoders; pointwise, pairwise, and listwise ranking; LLM reranking, diversity-aware context selection, cascades, latency, failure diagnosis, and implementation contracts.
32. **Advanced Retrieval and RAG Architectures.** Query rewriting, expansion, Multi-Query Retrieval, decomposition, HyDE, feedback-driven and multi-hop retrieval; hierarchical, recursive, and graph-based retrieval; adaptive routing, evidence aggregation, conflict handling, and implementation contracts.
33. **RAG Evaluation and Diagnostics.** Component and end-to-end evaluation; retrieval, context, generation, grounding, and citation metrics; ablations, error attribution, evaluation data, regression suites, online signals, cost, and implementation contracts.

### Parameter-Efficient Fine-Tuning

34. **Parameter-Efficient Fine-Tuning.** Full fine-tuning versus frozen-base adaptation; LoRA and QLoRA; Adapter, Prefix Tuning, and Prompt Tuning methods; trainable-state and memory accounting; merging and multi-adapter serving; quantized-base adaptation; failure modes; and implementation contracts. This chapter follows Chapter 12 conceptually and is published as a later peer chapter to preserve the established standalone sequence.

### Efficient Attention

35. **Efficient Attention and Head-Representation Design.** Attention IO and intermediate-state costs; tiled exact attention and online Softmax; FlashAttention and later execution improvements; MHA, MQA, GQA, and MLA as KV-representation designs; positional-encoding compatibility; KV Cache and Decode trade-offs; architectural versus kernel optimization; and implementation contracts. This chapter builds on Chapter 3's attention derivation, Chapter 20's KV Cache lifecycle, and Chapter 25's bottleneck analysis without changing their scope.

36. **Sparse and Long-Context Attention.** Structured attention connectivity; local, sliding-window, global, block-sparse, strided, dilated, and random patterns; Sparse Transformer, Longformer, and BigBird; receptive fields; decoder-only long-context constraints; KV Cache implications; RoPE extrapolation and context extension; retrieval versus long context; and implementation contracts. This chapter follows Chapter 35 by separating sparse architectural connectivity from FlashAttention-style dense execution.

## Planned directions

- **Agents and Tool Use:** tool interfaces, planning, execution loops, memory, environment interaction, and agent evaluation.
- **Multimodal Models:** vision-language inputs, multimodal tokenization and fusion, training objectives, evaluation, and system interfaces.

Each future chapter should declare its scope, prerequisites, and sources in its own directory and should remain readable without a combined-book build.
