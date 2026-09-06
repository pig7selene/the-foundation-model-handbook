# Codex Instructions for New Chapters

## Scope

These instructions govern every new standalone chapter in this repository. Chapters 1–25, `docs/style-guide.md`, `docs/chapter-template.md`, and the shared Typst files are the project baseline. Do not begin a new chapter by copying generic Foundation Model prose or by inventing a separate visual style.

The repository is English-only. Do not write Chinese explanatory prose, translated duplicate content, or a bilingual chapter structure.

## Required Workflow

### 1. Read the Existing Manuscript First

1. Check `git status` and preserve unrelated user changes.
2. Read the relevant earlier chapters, especially the immediate prerequisites and chapters that establish notation or implementation contracts.
3. Read `docs/style-guide.md`, `docs/chapter-template.md`, `docs/book-outline.md`, `README.md`, `templates/typst/chapter.typ`, `templates/typst/typography.typ`, `templates/typst/environments.typ`, `templates/typst/notation.typ`, and `references/handbook.bib`.
4. Inspect the current chapter source structure and the matching build filename before creating a new directory.

Do not rewrite, reorganize, or reformat published chapters unless a real consistency error or a necessary cross-reference correction is found. State the evidence for any such change.

### 2. Set Scope and Research

1. Confirm the chapter's place in `docs/book-outline.md`, its prerequisites, and its exclusions.
2. Identify the minimum set of authoritative sources needed to support consequential claims. Prefer original papers, peer-reviewed proceedings, standard textbooks, official reports, and documented dataset or model releases.
3. Add complete, clean entries to `references/handbook.bib`, using DOI fields when available and stable direct URLs otherwise.
4. Decide which concepts require a derivation, table, figure, formal environment, or implementation contract. Omit anything that does not improve understanding.

### 3. Write the Chapter

1. Create `chapters/part-name/NN-topic-slug/main.typ` and import the existing shared Typst template. Do not duplicate the template locally.
2. Use `technical-chapter` for the title page, `abstract` for the compact summary, `academic-table` for tables, and `chapter-bibliography` for references.
3. Follow the logical structure in `docs/chapter-template.md`, adapting it to the subject rather than padding it.
4. Define notation before use, state tensor shapes when they matter, and label equations, figures, tables, algorithms, definitions, and sections semantically.
5. Use earlier chapters as prerequisites. Refer to them in prose instead of duplicating their content.
6. Include `Implementation Contracts` when the topic has concrete shape, numerical, data, masking, provenance, or reproducibility invariants.
7. Keep all prose, captions, headings, and formal material in English.

### 4. Compile and Inspect

1. Build the standalone PDF:

   ```bash
   typst compile --root . chapters/part-name/NN-topic-slug/main.typ build/part-name/NN-topic-slug.pdf
   ```

2. Treat every compile error or warning as a problem to investigate. Verify bibliography entries, citations, labels, and imports.
3. Render the resulting PDF to images and visually inspect every page, with particular attention to the title page, dense prose pages, equations, tables or figures, section transitions, the summary, and references.
4. Check for overflow, clipped or crowded equations, bad table wrapping, awkward page breaks, excessive whitespace, broken numbering, spacing regressions, missing citations, unreadable bibliography entries, and incorrect page-number behavior.
5. Repair the source and repeat compilation and visual inspection until the latest PDF is clean.

### 5. Run Consistency Checks

1. Compare terminology and notation with the relevant completed chapters and `templates/typst/notation.typ`.
2. Confirm that technical conclusions do not contradict established chapters without an explicit, source-backed reason.
3. Verify that no earlier topic has been unnecessarily re-explained.
4. Confirm that the source, bibliography, README link, outline entry, generic `.gitignore` tracked-PDF rule, and `build/part-name/NN-topic-slug.pdf` agree on the chapter number, slug, and title.
5. Run `git diff --check` on source and documentation changes, then inspect `git status` to ensure only intended files will be committed.

### 6. Commit and Push

1. Stage the chapter source, the compiled PDF, new bibliography entries, and required index or build-metadata updates.
2. Use a concise descriptive commit message, for example `Add Pretraining Data chapter`.
3. Push the completed work to `origin/main`.
4. Verify that the local commit and `origin/main` resolve to the same revision and that the working tree is clean.

## Non-Negotiable Quality Bar

- The completed PDF is the artifact to review, not merely the Typst source.
- The visual language remains the shared restrained academic style: serif text, monochrome figures and tables, compact hierarchy, and no decorative UI elements.
- Assertions that depend on the literature are cited; citations are complete and readable.
- A chapter is complete only when its prose, notation, implementation consequences, references, rendered layout, repository metadata, commit, and remote state agree.
