# Style Guide

## Status and Authority

This guide records the conventions established by Chapters 1–15 and the shared Typst templates. It applies to every new standalone chapter. When this guide conflicts with an existing manuscript convention, the published chapters and `templates/typst/` are authoritative until the inconsistency is deliberately resolved across the project.

## Language and Tone

Write the entire manuscript in English. This includes titles, abstracts, headings, prose, definitions, captions, mathematical explanations, tables, notes, and references. Use standard research terminology in its established English form, including Transformer, Self-Attention, Causal Language Modeling, KV Cache, LayerNorm, RMSNorm, Pre-Norm, Post-Norm, RLHF, and PPO.

The prose should resemble a rigorous modern technical textbook or academic note. It should be concise, explanatory, and evidence-led. Begin a topic with its modeling or systems motivation, then make the formal object explicit, then explain its consequences for implementation, training, evaluation, or systems behavior. Prefer connected paragraphs to lists. State assumptions and scope conditions rather than turning empirical observations into universal laws.

Avoid a blog, tutorial, interview-preparation, marketing, slide-deck, or knowledge-base voice. Do not use conversational filler, generic claims of importance, artificial recaps, or long catalogues of definitions. Do not write Chinese prose, emoji, decorative icons, gradients, card-like layouts, or Web-documentation styling.

## Terminology and Notation

Use a term consistently after it has been introduced. Expand a less familiar abbreviation at first use; retain the conventional abbreviation afterward. Preserve meaningful distinctions such as token ID versus token embedding, logits versus probabilities, Attention Mask versus loss mask, and document versus sample versus sequence versus token.

The current shared mathematical vocabulary is:

- $cal(V)$ is the vocabulary and $V = |cal(V)|$ is its size.
- $x = (x_1, dots, x_T)$ is a token sequence; $t$ indexes token positions and $T$ is sequence length.
- $B$ is batch size, $d$ is model width, $L$ is the number of Transformer blocks, and $d_h$ is a head dimension when attention is in scope.
- $H in R^(B times T times d)$ is a batch-major residual-stream tensor; $h_t$ denotes a token-level hidden state when the batch axis is omitted.
- $theta$ denotes model parameters; $p_theta$ denotes a model distribution.
- $O$ denotes vocabulary logits; $y_(b,t)$ denotes a target token; $ell_(b,t)$ denotes a token-level loss; and $m_(b,t)$ denotes a loss mask when that distinction is needed.

Introduce every local symbol before use and state tensor shapes when a derivation or implementation contract depends on them. Use Typst mathematical conventions already present in the manuscript: $R^(...)$ for shapes, `times` for tensor axes, $cal(...)$ for sets, and $op("...")$ for named operators. `templates/typst/notation.typ` records the shared conventions and defines optional bold vector and matrix helpers; it is not imported automatically by `chapter.typ`, so import it explicitly only when a helper materially improves clarity. Record any genuinely global addition there before using it in a chapter. Do not introduce a second notation system.

## Sources and Citations

Ground consequential historical, technical, architectural, empirical, and systems claims in authoritative sources. Prefer original papers, peer-reviewed proceedings, standard textbooks, official technical reports, and carefully documented dataset or model releases. Do not use low-quality blogs as primary support.

Maintain the shared source database in `references/handbook.bib`. Add a complete BibTeX entry before citing a new work. Prefer a DOI when available; otherwise retain a stable, direct landing-page or report URL. Cite with `@citation-key` close to the claim it supports. Do not cite a source merely to decorate a paragraph, and do not leave unsupported claims that depend on a specific paper or reported result.

Each standalone chapter ends with the existing unnumbered `References` heading and `#chapter-bibliography("/references/handbook.bib")`. The shared template provides the compact reference typography; do not redefine bibliography styling locally.

## Structure and Headings

Chapters begin with a title, author/date metadata supplied by `technical-chapter`, a compact abstract, and a numbered `Introduction`. They proceed through a subject-driven sequence of descriptive sections and end with `Summary` followed by references. `Implementation Contracts` is a recurring and valuable section when a topic has executable assumptions, tensor invariants, numerical conditions, data schemas, or reproducibility requirements. It is not mandatory when it would add no practical value.

Use a level-1 Typst heading (`= Heading`) for the major logical units of a chapter. Use level-2 headings (`== Heading`) only where a section benefits from a genuine internal split, such as a derivation and its consequence. Level-3 headings are exceptional. Use descriptive Title Case headings, not vague headings such as "Discussion" or "More Details." Do not add sections merely to satisfy a fixed outline or to increase page count.

For concepts established in an earlier standalone chapter, refer to the relevant chapter and build on it rather than repeat its full derivation. Because chapters compile independently, cross-chapter references are written as prose such as "Chapter 5 explains ...". Use Typst cross-references within a chapter whenever a labeled equation, figure, table, definition, or section is being invoked.

## Equations, Tables, Figures, and Environments

Use display equations when formalization or a derivation materially improves understanding. Define notation first, give the display equation a semantic label immediately afterward, and explain the equation in prose. Equations are numbered automatically per chapter as `(1)`, `(2)`, and so on. Refer to them with `@eq-descriptive-name`; do not rely on page position or phrases such as "the equation above."

Use `#figure(academic-table(...), caption: [...])` for textbook-style tables. Tables should compare or clarify information that is harder to read in prose, have concise headers, and use a caption that states the reader's takeaway. Use figures for actual architecture, mechanism, or relationship diagrams, not for decoration. Keep all visual material monochrome, restrained, and legible at the template's body scale. Label tables as `tab-...` and figures as `fig-...`.

The shared environments define Definition, Theorem, Proposition, Lemma, Remark, Example, and Algorithm. Use them only when the formal distinction helps the reader. Chapter 1 uses Definition and Algorithm for a precise tokenizer contract and BPE procedure; later chapters rely mainly on prose and equations. Do not manufacture theorem-like statements for ordinary explanations.

Use topic-specific, semantic labels such as `sec-data-mixtures-sampling`, `eq-token-averaged-batch-loss`, `tab-kv-cache-accounting`, `fig-head-sharing`, `def-tokenizer`, and `alg-bpe-training`.

## Typography and Page Design

Use `templates/typst/chapter.typ` and do not override its page, font, paragraph, heading, header, footer, abstract, bibliography, or table settings in a chapter. The established design is a restrained English-language academic layout: serif body text, mathematical typesetting, A4 pages, generous but dense text blocks, justified paragraphs, compact numbered headings, a centered first-page title block, and centered page numbers after the first page.

The current template supplies the following project-wide design decisions:

- `technical-chapter` sets the document title, author `pig7selene`, date `September 5, 2026`, A4 margins, and first-page footer behavior.
- The body uses the shared serif fallback stack at 10.5 pt, justified paragraphs, no first-line indent, and restrained leading and paragraph spacing.
- Level-1, level-2, and level-3 headings are semibold, progressively smaller, and use aligned number/title grids with compact academic spacing.
- Tables use only horizontal rules; formal environments use a thin black left rule; figures and diagrams remain black and white.

Do not imitate Notion, Web documentation, presentation slides, callout-heavy study notes, or colorful educational cards. Do not add manual tabs, repeated literal spaces, decorative boxes, or local font changes to solve a structural layout problem. Adjust content, labels, table columns, or the shared template deliberately instead.

## Before Declaring a Chapter Complete

Check that the chapter has a clear scope, no avoidable repetition of earlier material, locally defined notation, accurate citations, semantic labels, and a conclusion that summarizes rather than repeats. Compile the standalone Typst source and inspect the final PDF visually for overflow, equation layout, tables, captions, page breaks, spacing, headers and footers, references, terminology, and notation. Correct problems before the chapter is committed and pushed.
