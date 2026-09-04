# The Foundation Model Handbook

*Architecture, Pretraining, Post-training, Reinforcement Learning, Inference, and Systems*

`The Foundation Model Handbook` is a long-lived personal technical handbook for the core knowledge needed in foundation-model algorithm work. Its destination is a carefully edited PDF with the character of an academic textbook or technical monograph, rather than a collection of Markdown notes.

## Goal

The handbook records understanding accumulated through actual study, implementation, and reading. It is deliberately not an automatically generated AI encyclopedia: material that has not yet been studied or verified remains a TODO.

## Current status

The repository currently contains the book architecture, Typst template, bibliography plumbing, and chapter placeholders. It intentionally does **not** yet contain substantive textbook chapters.

## Layout

```text
main.typ                 Book entry point
template/                Book, environment, and typography definitions
chapters/                Eight parts of the handbook, each with an index.typ skeleton
figures/                 Curated figures for the book
references/              BibTeX database
assets/                  Supporting non-figure assets
sources/                 Reading notes and source materials awaiting synthesis
build/                   Local compilation output (ignored except .gitkeep)
```

## Build

Install [Typst](https://typst.app/) and run:

```bash
typst compile main.typ build/the-foundation-model-handbook.pdf
```

The generated PDF is intentionally excluded from version control.

## Development principles

- Prefer precise, source-backed explanations to broad coverage.
- Preserve standard English technical terms such as Transformer, RLHF, KV Cache, and FSDP; use Chinese chiefly for explanation.
- Add equations, figures, examples, and references only when they clarify a studied concept.
- Keep unfinished material explicit with TODO markers rather than filling gaps with unverified prose.
- Maintain a restrained academic layout suitable for sustained reading and eventual print-quality PDF output.
