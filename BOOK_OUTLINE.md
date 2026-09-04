# Technical Chapter Roadmap

This repository publishes independent technical chapters rather than a single assembled book. The roadmap records a coherent learning sequence, but it does not create source directories until a chapter is ready to be researched, written, and visually reviewed.

## Published

1. **Tokenization and Input Representations.** Vocabulary design, reserved symbols, BPE, embedding lookup, positional inputs, token budgets, and implementation contracts.
2. **Transformer Architecture.** Decoder-only computation, residual streams, normalization placement, LM heads, shape contracts, and high-level resource costs.

## Planned directions

- **Architecture:** causal Self-Attention, masking, MHA/MQA/GQA, RoPE, FFNs, SwiGLU, LayerNorm, RMSNorm, initialization, and gradient flow.
- **Training:** Causal Language Modeling, data pipelines, deduplication, data mixing, AdamW, schedules, precision, scaling, stability, and checkpoints.
- **Post-training:** SFT, preference data, Reward Models, RL foundations, RLHF, PPO, DPO, GRPO, reasoning RL, and parameter-efficient adaptation.
- **Inference and systems:** decoding, KV Cache, quantization, speculative decoding, serving, GPU accounting, distributed training, communication, and fault recovery.
- **Evaluation and applications:** benchmark design, contamination, LLM-as-a-Judge, RAG, tool calling, agent systems, and agent evaluation.

Each future chapter should declare its scope, prerequisites, and sources in its own directory and should remain readable without a combined-book build.
