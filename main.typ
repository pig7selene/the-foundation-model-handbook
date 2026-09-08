#import "templates/typst/book.typ": handbook-book, part, chapter, book-header, book-footer

#show: handbook-book

#page(
  paper: "a4",
  margin: (top: 24mm, bottom: 25mm, left: 31mm, right: 27mm),
  header: none,
  footer: none,
)[
  #v(22%)
  #align(center)[
    #text(size: 28pt, weight: "semibold")[The Foundation Model Handbook]
    #v(1.55em)
    #text(size: 13pt)[Architecture, Pretraining, Post-training, Reinforcement Learning, Inference, and Systems]
    #v(5em)
    #text(size: 11pt)[pig7selene]
    #v(0.7em)
    #text(size: 10pt, fill: luma(50%))[Version 1.0 · September 2026]
  ]
]

#pagebreak()
#set page(header: none, footer: book-footer)
#align(center)[#text(size: 19pt, weight: "semibold")[Contents]]
#v(1em)
#outline(title: none, depth: 3)

#pagebreak()
#set page(header: book-header, footer: book-footer)

#part([I], [Foundations])
#chapter(1, 1, [Tokenization and Input Representations])[
  #include "chapters/foundations/01-tokenization-and-input-representations/main.typ"
]

#part([II], [Architecture])
#chapter(2, 2, [Transformer Architecture])[
  #include "chapters/architecture/02-transformer-architecture/main.typ"
]
#chapter(2, 3, [Attention and Position Encoding])[
  #include "chapters/architecture/03-attention-and-position-encoding/main.typ"
]
#chapter(2, 4, [Feed-Forward Networks, Normalization, and Residual Connections])[
  #include "chapters/architecture/04-feed-forward-normalization-and-residual-connections/main.typ"
]

#part([III], [Pretraining])
#chapter(3, 5, [Pretraining Objective and Language Modeling])[
  #include "chapters/pretraining/05-pretraining-objective-and-language-modeling/main.typ"
]
#chapter(3, 6, [Pretraining Data])[
  #include "chapters/pretraining/06-pretraining-data/main.typ"
]
#chapter(3, 7, [Optimization for Pretraining])[
  #include "chapters/pretraining/07-optimization-for-pretraining/main.typ"
]
#chapter(3, 8, [Numerical Precision and Training Stability])[
  #include "chapters/pretraining/08-numerical-precision-and-training-stability/main.typ"
]
#chapter(3, 9, [Scaling Laws and Compute])[
  #include "chapters/pretraining/09-scaling-laws-and-compute/main.typ"
]
#chapter(3, 10, [Distributed Training])[
  #include "chapters/pretraining/10-distributed-training/main.typ"
]
#chapter(3, 11, [Evaluation, Checkpointing, and Training Diagnostics])[
  #include "chapters/pretraining/11-evaluation-checkpointing-and-training-diagnostics/main.typ"
]
#chapter(3, 12, [Practical FSDP and Distributed Training State])[
  #include "chapters/pretraining/12-practical-fsdp-and-distributed-training-state/main.typ"
]

#part([IV], [Post-training, Alignment, and Adaptation])
#chapter(4, 13, [Supervised Fine-Tuning])[
  #include "chapters/post-training/13-supervised-fine-tuning/main.typ"
]
#chapter(4, 14, [Parameter-Efficient Fine-Tuning])[
  #include "chapters/parameter-efficient-fine-tuning/14-parameter-efficient-fine-tuning/main.typ"
]
#chapter(4, 15, [Preference Data and Reward Modeling])[
  #include "chapters/post-training/15-preference-data-and-reward-modeling/main.typ"
]
#chapter(4, 16, [RLHF and PPO])[
  #include "chapters/post-training/16-rlhf-and-ppo/main.typ"
]
#chapter(4, 17, [Direct Preference Optimization])[
  #include "chapters/post-training/17-direct-preference-optimization/main.typ"
]
#chapter(4, 18, [Group Relative Policy Optimization])[
  #include "chapters/post-training/18-group-relative-policy-optimization/main.typ"
]
#chapter(4, 19, [Reasoning RL, Rollouts, and Verifiable Rewards])[
  #include "chapters/post-training/19-reasoning-rl-rollouts-and-verifiable-rewards/main.typ"
]
#chapter(4, 20, [Post-Training Evaluation and Alignment Trade-offs])[
  #include "chapters/post-training/20-post-training-evaluation-and-alignment-trade-offs/main.typ"
]
#chapter(4, 21, [On-Policy Alignment and Iterative Policy Improvement])[
  #include "chapters/post-training/21-on-policy-alignment-and-iterative-policy-improvement/main.typ"
]
#chapter(4, 22, [Distributed RL Training Systems])[
  #include "chapters/post-training/22-distributed-rl-training-systems/main.typ"
]

#part([V], [Inference and Serving])
#chapter(5, 23, [LLM Inference Fundamentals])[
  #include "chapters/inference-serving/23-llm-inference-fundamentals/main.typ"
]
#chapter(5, 24, [KV Cache and Memory Optimization])[
  #include "chapters/inference-serving/24-kv-cache-and-memory-optimization/main.typ"
]
#chapter(5, 25, [Quantization for LLM Inference])[
  #include "chapters/inference-serving/25-quantization-for-llm-inference/main.typ"
]
#chapter(5, 26, [Batching, Scheduling, and LLM Serving Systems])[
  #include "chapters/inference-serving/26-batching-scheduling-and-llm-serving-systems/main.typ"
]
#chapter(5, 27, [Speculative Decoding and Inference Acceleration])[
  #include "chapters/inference-serving/27-speculative-decoding-and-inference-acceleration/main.typ"
]
#chapter(5, 28, [Distributed LLM Inference and Parallelism])[
  #include "chapters/inference-serving/28-distributed-llm-inference-and-parallelism/main.typ"
]
#chapter(5, 29, [Inference System Design and Performance Optimization])[
  #include "chapters/inference-serving/29-inference-system-design-and-performance-optimization/main.typ"
]

#part([VI], [Efficient Attention and Long Context])
#chapter(6, 30, [Efficient Attention and Head-Representation Design])[
  #include "chapters/efficient-attention/30-efficient-attention-and-head-representation-design/main.typ"
]
#chapter(6, 31, [Sparse and Long-Context Attention])[
  #include "chapters/efficient-attention/31-sparse-and-long-context-attention/main.typ"
]

#part([VII], [Retrieval-Augmented Generation and Knowledge Augmentation])
#chapter(7, 32, [Retrieval-Augmented Generation Fundamentals])[
  #include "chapters/rag-knowledge-augmentation/32-retrieval-augmented-generation-fundamentals/main.typ"
]
#chapter(7, 33, [Embeddings and Semantic Retrieval])[
  #include "chapters/rag-knowledge-augmentation/33-embeddings-and-semantic-retrieval/main.typ"
]
#chapter(7, 34, [Vector Search and Approximate Nearest Neighbors])[
  #include "chapters/rag-knowledge-augmentation/34-vector-search-and-approximate-nearest-neighbors/main.typ"
]
#chapter(7, 35, [Chunking and Document Segmentation])[
  #include "chapters/rag-knowledge-augmentation/35-chunking-and-document-segmentation/main.typ"
]
#chapter(7, 36, [Sparse Retrieval and Hybrid Search])[
  #include "chapters/rag-knowledge-augmentation/36-sparse-retrieval-and-hybrid-search/main.typ"
]
#chapter(7, 37, [Reranking and Retrieval Refinement])[
  #include "chapters/rag-knowledge-augmentation/37-reranking-and-retrieval-refinement/main.typ"
]
#chapter(7, 38, [Advanced Retrieval and RAG Architectures])[
  #include "chapters/rag-knowledge-augmentation/38-advanced-retrieval-and-rag-architectures/main.typ"
]
#chapter(7, 39, [RAG Evaluation and Diagnostics])[
  #include "chapters/rag-knowledge-augmentation/39-rag-evaluation-and-diagnostics/main.typ"
]
