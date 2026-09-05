# Chapter Template

## Purpose

This is the usual logical and Typst structure for a new standalone chapter. It is derived from Chapters 1–6, not imposed as a rigid outline. Adapt the sequence, depth, and number of sections to the subject. Do not add empty sections, a formal environment, a figure, or an `Implementation Contracts` section merely because this template contains an option for it.

Every chapter must remain intelligible as an independent PDF while connecting explicitly to relevant earlier chapters.

## Minimal Typst Skeleton

```typst
#import "../../../templates/typst/chapter.typ": technical-chapter, abstract, academic-table, chapter-bibliography
// Import only the environments required by this chapter.
// #import "../../../templates/typst/environments.typ": definition, algorithm

#show: technical-chapter.with(
  title: [Chapter Title],
)

#abstract[
  State the technical object, the main conceptual path, and the practical or
  mathematical consequence in one compact paragraph.
]

= Introduction <sec-topic-introduction>

Motivate the topic, state its scope, and connect it to the existing manuscript.
Define only the global notation needed before the first derivation.

= Core Concept or System Model <sec-topic-core-model>

Develop the central object through prose, equations, diagrams, or a table as needed.

== A Necessary Internal Distinction <sec-topic-distinction>

Use a subsection only when it gives the reader a real structural aid.

= Variants, Trade-offs, or Consequences <sec-topic-consequences>

Explain the relationship to implementation, training, inference, evaluation, or systems behavior.

= Implementation Contracts <sec-topic-implementation-contracts>

Include only when the topic has testable interfaces, shape constraints, numerical requirements,
data invariants, or reproducibility conditions.

= Summary <sec-topic-summary>

Summarize the chapter's resolved conceptual relationships. Do not introduce new material.

// Use when it gives the references a clean opening page.
#pagebreak()
#heading(level: 1, numbering: none, outlined: false)[References]
#chapter-bibliography("/references/handbook.bib")
```

The `#pagebreak()` before references is optional. Chapters 5 and 6 use it to give the bibliography a clean start; Chapters 1–4 allow references to follow the final prose directly. Preserve the choice that gives the particular chapter the cleaner ending.

## Usual Logical Progression

1. **Introduction.** Explain why the topic is needed in the decoder-only language-model stack, data pipeline, or training process. State the chapter's boundary and identify earlier chapters that supply prerequisites.
2. **Core model.** Define the objects, variables, tensor shapes, or system interfaces. Give intuition before or alongside formalism.
3. **Derivation or mechanism.** Derive equations, algorithms, data transformations, or execution behavior only to the depth required by the chapter's learning objective. State assumptions and explain what each term means.
4. **Variants and trade-offs.** Compare the alternatives that matter in practice, such as Pre-Norm versus Post-Norm, MHA versus GQA, exact versus near deduplication, or token inventory versus drawn-token budget.
5. **Consequences.** Connect the mechanism to compute, memory, numerical stability, data distribution, evaluation validity, or model behavior.
6. **Implementation Contracts, when appropriate.** State the invariants that convert the mathematical description into a reliable tensor program or data pipeline.
7. **Summary.** Reconnect the central object, formal result, and operational consequence without repeating the chapter section by section.

This order is illustrative. A data chapter can proceed from source provenance to sequence construction; an architecture chapter can begin from tensor shapes and then derive a mechanism; an evaluation chapter may place its threat model before its metric. Preserve the logic of the subject, not the number of headings.

## Content Rules

- Start with motivation, then introduce notation, then formalize the topic.
- Use continuous exposition as the default. A short list is appropriate for a compact contract, comparison, or checklist, but not as a substitute for explanation.
- Refer to prior standalone chapters instead of restating settled fundamentals. Since the PDFs compile independently, name prior chapters in prose rather than attempting a cross-file Typst reference.
- Give a display equation a semantic label immediately after it and explain its role afterward.
- Use `academic-table` for comparison tables and `#figure` for numbered tables or diagrams. Every caption should add meaning beyond the title.
- Use Definition, Algorithm, Theorem, Proposition, Lemma, Remark, or Example only when the named structure improves the exposition.
- Cite source-backed claims in the shared BibTeX database; retain citations only where they support the chapter's argument.
- Do not write toward a target page count. Depth follows the topic's prerequisites and the learning objective.

## Recommended Directory and Build Convention

Create a chapter directory only when the chapter is ready to be researched and written. Place it beneath its conceptual part:

```text
chapters/part-name/NN-topic-slug/
└── main.typ
```

Build it independently:

```bash
typst compile --root . chapters/part-name/NN-topic-slug/main.typ build/part-name/NN-topic-slug.pdf
```

Version the generated PDF in `build/`, confirm that the generic tracked-PDF rule in `.gitignore` covers its new part directory, add a README link under the correct topic group, and update `docs/book-outline.md` once the chapter is complete.
