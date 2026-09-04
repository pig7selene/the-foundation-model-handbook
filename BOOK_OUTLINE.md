# Detailed Table of Contents

This document is the canonical outline for *The Foundation Model Handbook*. It is intentionally a plan rather than textbook prose. The chapter files retain only TODO placeholders until a chapter is written from first principles and supported by primary sources.

## Scope and depth conventions

The book is aimed at an undergraduate or graduate reader preparing for a Foundation Model / LLM Algorithm Internship. It is deliberately selective: every chapter must materially help the reader understand, train, optimize, evaluate, or implement a modern Foundation Model.

Expected depth uses three practical levels:

- **Derive and implement**: derive the central equations, trace tensor shapes and costs, and implement a minimal correct version.
- **Operate and diagnose**: calculate the relevant resource or optimization quantities and diagnose common failures in a real training or serving stack.
- **Explain and compare**: explain assumptions, trade-offs, and when a method should or should not be chosen; detailed systems APIs are out of scope.

## Part I — Foundations

### Chapter 1 — Tokenization and Input Representations

- **Learning objective:** Explain how text becomes model inputs and why tokenization choices affect data, compute, and evaluation.
- **Prerequisites:** Basic discrete mathematics and linear algebra.
- **Core concepts:** vocabulary construction, BPE, token boundaries, special tokens, token embeddings, positional representations, sequence length, token counts.
- **Expected depth:** Derive and implement a small BPE-style vocabulary construction loop; reason about token-budget consequences.

### Chapter 2 — Causal Language Modeling

- **Learning objective:** Derive the autoregressive objective used by decoder-only Foundation Models.
- **Prerequisites:** Chapter 1; probability, logarithms, and expectation.
- **Core concepts:** chain-rule factorization, next-token prediction, teacher forcing, shifted labels, negative log-likelihood, cross entropy, perplexity, train/evaluation mismatch.
- **Expected depth:** Derive and implement the causal language-model loss, including its masking and normalization conventions.

### Chapter 3 — The Transformer Computation Graph

- **Learning objective:** Trace one decoder-only Transformer block from embeddings to logits.
- **Prerequisites:** Chapters 1–2; matrix multiplication and basic neural networks.
- **Core concepts:** residual connection, pre-norm design, hidden states, projection layers, tied embeddings, depth, width, parameter counting.
- **Expected depth:** Derive and implement a minimal decoder-only Transformer with correct tensor shapes.

### Chapter 4 — Self-Attention and Causal Masking

- **Learning objective:** Derive scaled dot-product attention and explain why the Attention Mask is both a modeling and systems contract.
- **Prerequisites:** Chapter 3; linear algebra and softmax.
- **Core concepts:** queries, keys, values, scaling, causal Attention Mask, padding masks, softmax, numerical stability, attention complexity.
- **Expected depth:** Derive the forward and backward-relevant quantities, and implement masked attention without leakage or unstable softmax.

### Chapter 5 — Attention Variants and Position Encoding

- **Learning objective:** Compare MHA, MQA, GQA, and RoPE through representation capacity, KV-cache cost, and decoding throughput.
- **Prerequisites:** Chapter 4.
- **Core concepts:** attention heads, grouped sharing, RoPE, relative position, context extrapolation, KV-head count.
- **Expected depth:** Explain and calculate the memory and communication implications of each variant; implement the basic variants.

### Chapter 6 — Feed-Forward Networks, Normalization, and Gradient Flow

- **Learning objective:** Explain the non-attention half of a Transformer and the architectural choices that keep deep networks trainable.
- **Prerequisites:** Chapters 3–4; multivariable calculus.
- **Core concepts:** FFN, SwiGLU, LayerNorm, RMSNorm, Weight Initialization, residual scale, Gradient Flow, activation statistics, numerical stability.
- **Expected depth:** Derive the main normalization and gating operations, then diagnose exploding, vanishing, or poorly scaled activations.

## Part II — Pretraining

### Chapter 7 — Pretraining Data Pipelines

- **Learning objective:** Map raw corpora to deterministic, auditable token streams suitable for large-scale Causal Language Modeling.
- **Prerequisites:** Chapters 1–2; basic data processing.
- **Core concepts:** collection, licensing awareness, normalization, filtering, document packing, sharding, streaming, reproducibility, data provenance.
- **Expected depth:** Operate and diagnose a small-scale data pipeline; understand which decisions change the effective training distribution.

### Chapter 8 — Data Quality, Deduplication, Mixing, and Curriculum

- **Learning objective:** Reason about data as an optimization variable rather than a fixed input.
- **Prerequisites:** Chapter 7; elementary statistics.
- **Core concepts:** exact and near Deduplication, quality filters, domain mixtures, Data Curriculum, reweighting, contamination risk, mixture schedules.
- **Expected depth:** Explain and compare practical policies; design a defensible small-scale mixture without claiming universal optimality.

### Chapter 9 — Optimizers and Optimizer State

- **Learning objective:** Derive AdamW and account for the state it adds to a training system.
- **Prerequisites:** Chapters 2 and 6; gradients and optimization.
- **Core concepts:** SGD, momentum, adaptive moments, bias correction, decoupled weight decay, Optimizer State, parameter groups.
- **Expected depth:** Derive AdamW updates and calculate optimizer-memory costs; implement and inspect a training step.

### Chapter 10 — Learning-Rate Schedules, Batches, and Token Budgets

- **Learning objective:** Connect Learning Rate, Warmup, global batch size, sequence length, and total Token Budget into one training plan.
- **Prerequisites:** Chapter 9; basic optimization.
- **Core concepts:** Learning Rate Scheduler, Warmup, batch/sequence trade-offs, gradient noise, effective batch, tokens per update, context-length curriculum.
- **Expected depth:** Operate and diagnose schedule and batch choices using token-based, rather than epoch-based, accounting.

### Chapter 11 — Mixed Precision and Numerical Stability

- **Learning objective:** Explain why large-model training needs Mixed Precision and where reduced precision can silently fail.
- **Prerequisites:** Chapters 4, 6, and 9; floating-point arithmetic.
- **Core concepts:** FP32, FP16, BF16, loss scaling, overflow, underflow, accumulation precision, stable reductions, precision boundaries.
- **Expected depth:** Operate and diagnose a mixed-precision loop, including non-finite loss and gradient failures.

### Chapter 12 — Scaling Laws and Compute Planning

- **Learning objective:** Use scaling-law reasoning to relate model size, data quantity, and compute budget before a training run.
- **Prerequisites:** Chapters 7–10; logarithms and basic regression intuition.
- **Core concepts:** FLOPs estimation, compute budget, parameter/data scaling, Chinchilla-style scaling, Token Budget, model-data trade-offs.
- **Expected depth:** Calculate first-order training compute and use scaling laws as planning heuristics, not as immutable laws.

### Chapter 13 — Training Dynamics, Stability, and Recovery

- **Learning objective:** Recognize and recover from the recurrent failure modes of large-scale pretraining.
- **Prerequisites:** Chapters 9–12.
- **Core concepts:** Training Dynamics, Gradient Accumulation, Gradient Clipping, loss spikes, gradient norms, divergence, monitoring, Checkpoint, Checkpoint Recovery, validation loss.
- **Expected depth:** Operate and diagnose a fault-tolerant training loop; form evidence-based hypotheses for loss spikes and failed resumes.

## Part III — Post-training & Alignment

### Chapter 14 — Supervised Fine-Tuning

- **Learning objective:** Formulate SFT as conditional language modeling and understand its data and formatting constraints.
- **Prerequisites:** Chapters 2, 7, and 9.
- **Core concepts:** instruction-response data, chat templates, loss masking, multi-turn data, distribution shift, quality control, overfitting.
- **Expected depth:** Derive and implement an SFT loss and evaluate a small fine-tuning run.

### Chapter 15 — Preference Data and Reward Models

- **Learning objective:** Turn comparisons into a learned reward signal while understanding its uncertainty and biases.
- **Prerequisites:** Chapter 14; probability and binary classification.
- **Core concepts:** preference pairs, Bradley–Terry-style modeling, Reward Model, calibration, reward generalization, annotator disagreement, Reward Evaluation.
- **Expected depth:** Explain and compare reward-model design choices; implement the central pairwise objective.

### Chapter 16 — Reinforcement Learning Fundamentals for Language Models

- **Learning objective:** Establish the RL vocabulary required for policy optimization of a language model.
- **Prerequisites:** Chapters 2 and 14; probability and expectation.
- **Core concepts:** trajectories, return, policy gradient, value function, Actor-Critic, On-policy and Off-policy learning, exploration, credit assignment.
- **Expected depth:** Derive the policy-gradient estimator and explain why LLM RL has unusual sequence-level costs.

### Chapter 17 — Rollouts, Advantages, and GAE

- **Learning objective:** Build the bridge from sampled completions to low-variance policy-gradient updates.
- **Prerequisites:** Chapter 16.
- **Core concepts:** Rollout, reward assignment, baselines, Advantage Estimation, Generalized Advantage Estimation (GAE), variance-bias trade-off, masking variable-length trajectories.
- **Expected depth:** Derive GAE and implement a correct rollout-to-batch transformation.

### Chapter 18 — RLHF and PPO

- **Learning objective:** Explain RLHF as constrained policy improvement and derive the practical PPO objective.
- **Prerequisites:** Chapters 15–17.
- **Core concepts:** reference policy, KL Regularization, PPO clipping, value loss, entropy, reward normalization, rollout/update ratio, policy drift.
- **Expected depth:** Derive the loss terms and operate a minimal PPO-style training loop with meaningful diagnostics.

### Chapter 19 — Direct and Group Preference Optimization

- **Learning objective:** Compare DPO, GRPO, and DAPO as alternatives to reward-model-plus-PPO pipelines.
- **Prerequisites:** Chapters 15–18.
- **Core concepts:** DPO objective, implicit reward, group-relative normalization, GRPO, DAPO, offline versus on-policy preference learning, objective assumptions.
- **Expected depth:** Derive the core DPO objective; explain the operational trade-offs of direct and group-relative methods.

### Chapter 20 — Reward Failure Modes and Reasoning RL

- **Learning objective:** Analyze why optimizing a reward can degrade useful behavior and why Verifiable Reward changes the design space for reasoning.
- **Prerequisites:** Chapters 15–19.
- **Core concepts:** Reward Hacking, Length Bias, reward misspecification, KL trade-offs, Verifiable Reward, Reasoning RL, Expert Iteration, rejection sampling.
- **Expected depth:** Explain and diagnose failure modes; design a source-backed evaluation plan for a verifiable-reward task.

## Part IV — Parameter-Efficient Fine-Tuning

### Chapter 21 — PEFT Principles and LoRA

- **Learning objective:** Explain why low-rank adaptation can approximate useful parameter updates with far fewer trainable weights.
- **Prerequisites:** Chapters 3, 6, and 14; linear algebra.
- **Core concepts:** full fine-tuning baseline, low-rank update, LoRA rank and scaling, target modules, initialization, adapter merging.
- **Expected depth:** Derive the LoRA parameterization and implement it for a linear projection.

### Chapter 22 — QLoRA and Quantized Fine-Tuning

- **Learning objective:** Connect quantized base weights with trainable low-rank adapters and memory-aware optimization.
- **Prerequisites:** Chapters 11 and 21.
- **Core concepts:** quantization error, NF4-style quantization, double quantization, paged optimizers, QLoRA memory budget.
- **Expected depth:** Operate and diagnose a constrained-memory fine-tuning run; calculate the principal memory trade-offs.

### Chapter 23 — Adapters and Soft-Prompt Methods

- **Learning objective:** Compare Adapter, Prefix Tuning, and Prompt Tuning with LoRA under a common adaptation framework.
- **Prerequisites:** Chapters 3–6 and 21.
- **Core concepts:** bottleneck adapters, virtual tokens, Prefix Tuning, Prompt Tuning, parameter isolation, task transfer, deployment constraints.
- **Expected depth:** Explain and compare methods; select an approach for a stated resource and deployment constraint.

## Part V — Inference

### Chapter 24 — Autoregressive Decoding and Sampling

- **Learning objective:** Turn next-token probabilities into controllable generation behavior.
- **Prerequisites:** Chapter 2; probability.
- **Core concepts:** greedy decoding, Temperature, Top-k, Top-p, repetition controls, Beam Search, length normalization, sampling reproducibility.
- **Expected depth:** Implement a decoding loop and explain how each policy changes quality, diversity, and latency.

### Chapter 25 — KV Cache and Decode-Time Memory

- **Learning objective:** Derive why KV Cache is essential for efficient autoregressive decoding and how it shapes serving capacity.
- **Prerequisites:** Chapters 4–5 and 24.
- **Core concepts:** prefill versus decode, cache layout, sequence growth, KV-heads, memory accounting, context window, cache eviction.
- **Expected depth:** Calculate KV-cache memory and implement cache-aware decoding.

### Chapter 26 — Efficient Attention at Inference

- **Learning objective:** Explain how attention is made IO-efficient or structurally cheaper without changing the user-facing model contract.
- **Prerequisites:** Chapters 4–5 and 25.
- **Core concepts:** FlashAttention, tiling, memory bandwidth, Sparse Attention, Multi-head Latent Attention (MLA), quality–efficiency trade-offs.
- **Expected depth:** Explain and compare mechanisms; estimate which bottleneck each method addresses.

### Chapter 27 — Inference Quantization

- **Learning objective:** Choose and evaluate quantization schemes for weights, activations, and KV Cache.
- **Prerequisites:** Chapters 11 and 25.
- **Core concepts:** calibration, per-channel scales, weight-only versus weight-activation quantization, KV-cache quantization, accuracy–latency–memory trade-off.
- **Expected depth:** Operate and diagnose an inference-quantization workflow; report quality regressions responsibly.

### Chapter 28 — Speculative Decoding

- **Learning objective:** Derive draft-and-verify generation and identify when it improves end-to-end latency.
- **Prerequisites:** Chapters 24–25; probability.
- **Core concepts:** draft model, target model, acceptance correction, token acceptance rate, batching interaction, latency model.
- **Expected depth:** Derive correctness at a high level and implement a minimal speculative-decoding loop.

### Chapter 29 — LLM Serving and Continuous Batching

- **Learning objective:** Understand how a serving engine converts variable request streams into high-throughput GPU work.
- **Prerequisites:** Chapters 25–28.
- **Core concepts:** PagedAttention, Continuous Batching, scheduler policies, prefill/decode separation, queueing, vLLM architecture, throughput versus tail latency.
- **Expected depth:** Operate and diagnose a serving stack; calculate the main capacity and latency constraints.

## Part VI — LLM Systems

### Chapter 30 — Performance and Memory Accounting

- **Learning objective:** Account for the memory and compute costs that determine whether a training configuration fits and performs well.
- **Prerequisites:** Chapters 3, 9, 11, and 12.
- **Core concepts:** GPU Memory, parameters, gradients, Optimizer State, activations, FLOPs, throughput, Model FLOPs Utilization (MFU), arithmetic intensity.
- **Expected depth:** Calculate a first-order memory budget and MFU; interpret utilization measurements without overclaiming precision.

### Chapter 31 — Distributed Training Fundamentals and Communication

- **Learning objective:** Explain how collective communication constrains distributed LLM training.
- **Prerequisites:** Chapter 30; basic parallel computing.
- **Core concepts:** process groups, topology, bandwidth, latency, All-Reduce, All-Gather, Reduce-Scatter, Communication Cost, overlap.
- **Expected depth:** Explain and calculate communication volumes for common collectives; identify likely bottlenecks from a training trace.

### Chapter 32 — Data Parallelism, ZeRO, and FSDP

- **Learning objective:** Compare replication and sharding strategies for parameters, gradients, and Optimizer State.
- **Prerequisites:** Chapters 9, 30, and 31.
- **Core concepts:** Data Parallel, gradient synchronization, ZeRO stages, FSDP, parameter all-gather, reduce-scatter, memory–communication exchange.
- **Expected depth:** Operate and diagnose a sharded data-parallel job; calculate memory savings and communication consequences.

### Chapter 33 — Tensor Parallelism and Sequence Parallelism

- **Learning objective:** Partition Transformer computation across devices while preserving mathematically equivalent results.
- **Prerequisites:** Chapters 3–6 and 31.
- **Core concepts:** Tensor Parallel, row/column parallel linear layers, attention partitioning, Sequence Parallel, activation partitioning, synchronization points.
- **Expected depth:** Trace tensor shapes and collective operations through one Transformer layer.

### Chapter 34 — Pipeline and Hybrid Parallelism

- **Learning objective:** Compose multiple parallelism axes into a feasible large-model training configuration.
- **Prerequisites:** Chapters 32–33.
- **Core concepts:** Pipeline Parallel, microbatches, pipeline bubble, schedules, three-dimensional parallelism, load balance, fault domains.
- **Expected depth:** Explain and compare schedules; choose a hybrid-parallel configuration from model and cluster constraints.

### Chapter 35 — Training Infrastructure and Distributed Recovery

- **Learning objective:** Treat training as a recoverable distributed system rather than a single long-running process.
- **Prerequisites:** Chapters 13 and 30–34.
- **Core concepts:** Distributed Checkpoint, resharding, Checkpoint Recovery, deterministic resume, orchestration, Ray, veRL, DataProto.
- **Expected depth:** Operate and diagnose recovery workflows; explain these tools as architectural patterns rather than memorize APIs.

## Part VII — Evaluation

### Chapter 36 — Language-Model Evaluation Fundamentals

- **Learning objective:** Interpret intrinsic metrics and build an evaluation protocol that matches a model claim.
- **Prerequisites:** Chapter 2; elementary statistics.
- **Core concepts:** held-out loss, Perplexity, calibration, variance, confidence intervals, prompt sensitivity, reproducibility.
- **Expected depth:** Calculate and interpret perplexity; design a small, statistically honest evaluation run.

### Chapter 37 — Benchmark Design and Data Contamination

- **Learning objective:** Judge whether a benchmark score is evidence of capability rather than evidence of leaked data or a weak protocol.
- **Prerequisites:** Chapters 7–8 and 36.
- **Core concepts:** Benchmarking, task validity, split design, contamination pathways, decontamination, leakage audits, reporting protocol.
- **Expected depth:** Explain and compare contamination controls; design a benchmark report with credible limitations.

### Chapter 38 — Preference, Reward, and Judge-Based Evaluation

- **Learning objective:** Evaluate aligned behavior when an automatic exact-match metric is unavailable.
- **Prerequisites:** Chapters 15 and 36–37.
- **Core concepts:** LLM-as-a-Judge, pairwise comparison, human evaluation, Reward Evaluation, judge calibration, positional bias, length bias, agreement.
- **Expected depth:** Operate and diagnose a judge-based protocol; distinguish a useful proxy from a ground-truth claim.

### Chapter 39 — Reasoning Evaluation

- **Learning objective:** Evaluate reasoning behavior without conflating answer accuracy, process quality, and benchmark contamination.
- **Prerequisites:** Chapters 20 and 36–38.
- **Core concepts:** reasoning traces, outcome versus process supervision, verifiable tasks, pass@k, tool use, robustness, adversarial variants.
- **Expected depth:** Design and critique a reasoning-evaluation suite with clear failure analysis.

## Part VIII — Applications

### Chapter 40 — Prompting and In-Context Learning

- **Learning objective:** Treat prompts as task interfaces that can be designed, tested, and versioned.
- **Prerequisites:** Chapters 2 and 24; evaluation basics.
- **Core concepts:** instruction hierarchy, few-shot prompting, delimiters, structured outputs, context management, prompt sensitivity, prompt evaluation.
- **Expected depth:** Implement and compare prompt templates under a fixed evaluation protocol.

### Chapter 41 — Retrieval-Augmented Generation

- **Learning objective:** Design RAG systems whose retrieval and generation components can be evaluated separately.
- **Prerequisites:** Chapters 7, 36, and 40.
- **Core concepts:** chunking, embedding retrieval, reranking, context construction, grounding, retrieval recall, citation faithfulness, RAG evaluation.
- **Expected depth:** Implement a minimal RAG pipeline and diagnose whether an error originates in retrieval, context, or generation.

### Chapter 42 — Tool Calling

- **Learning objective:** Connect model outputs to typed external actions without hiding reliability and safety constraints.
- **Prerequisites:** Chapters 24, 36, and 40.
- **Core concepts:** schema-constrained output, function calling, tool selection, argument validation, execution feedback, retries, idempotence, sandbox boundaries.
- **Expected depth:** Implement a small tool-calling loop and evaluate its success, latency, and failure modes.

### Chapter 43 — Agent Systems

- **Learning objective:** Analyze an Agent as a controlled loop of observation, planning, action, memory, and verification.
- **Prerequisites:** Chapters 39–42.
- **Core concepts:** planning, ReAct-style loops, short- and long-term memory, state, tool orchestration, stopping criteria, reliability, framework abstractions such as LangChain.
- **Expected depth:** Explain and compare agent designs; implement a narrow, observable agent loop rather than a general autonomous system.

### Chapter 44 — Agent Evaluation

- **Learning objective:** Evaluate Agents as systems whose trajectories, tool effects, cost, and safety matter alongside final answers.
- **Prerequisites:** Chapters 37–39 and 42–43.
- **Core concepts:** task suites, trajectory evaluation, environment design, reproducibility, cost and latency, intervention tests, safety evaluation, regression testing.
- **Expected depth:** Design an agent-evaluation harness with measurable success criteria and actionable failure categories.

## Dependency map

The reading order is intentionally a dependency order, not merely a taxonomy:

```text
Part I: representation + causal objective + Transformer mechanics
  -> Part II: data + optimization + scaling + stable pretraining
  -> Part III: SFT + preferences + RL objectives + reasoning alignment
  -> Part IV: parameter-efficient adaptation
  -> Part V: decoding + cache + efficient serving
  -> Part VI: distributed training and recovery
  -> Part VII: evaluation methodology and contamination controls
  -> Part VIII: prompt, RAG, tools, and agents
```

Three cross-cutting dependencies deserve special attention. Chapter 2 supplies the likelihood and sampling vocabulary reused by SFT, decoding, and evaluation. Chapters 9–13 supply the optimization, precision, and recovery vocabulary reused by post-training and distributed systems. Chapters 36–39 are required before application claims are treated as evidence: applications are evaluated systems, not demonstrations.

## Deliberate boundaries

The outline excludes topics without a direct Foundation Model internship payoff, exhaustive framework API tutorials, broad AI history, and encyclopedic application catalogs. Multimodal Foundation Models, Mixture-of-Experts, long-context training, safety policy, and retrieval internals are intentionally deferred until a concrete learning goal requires them; they can become focused future editions rather than diluting the first complete volume.
