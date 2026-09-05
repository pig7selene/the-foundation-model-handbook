#let body-font = ("Libertinus Serif", "New Computer Modern", "Times New Roman")

#let numbered-heading(number, title, number-width: 1.4em) = grid(
  columns: (number-width, 1fr),
  column-gutter: 0pt,
  align: (left, left),
  [#number],
  [#title],
)

#let setup-chapter-typography(body) = {
  set text(font: body-font, size: 10.5pt, lang: "en")
  set par(
    justify: true,
    leading: 0.68em,
    first-line-indent: 0pt,
    spacing: 0.7em,
  )
  set heading(numbering: "1.1")
  set math.equation(numbering: "(1)")

  show heading.where(level: 1): it => block(
    width: 100%,
    above: 1.8em,
    below: 0.45em,
  )[
    #align(left)[
      #text(size: 15.5pt, weight: "semibold")[
        #if it.numbering == none [#it.body] else [
          #numbered-heading(counter(heading).display(it.numbering), it.body, number-width: 1.4em)
        ]
      ]
    ]
  ]

  show heading.where(level: 2): it => block(
    width: 100%,
    above: 1.3em,
    below: 0.3em,
  )[
    #align(left)[
      #text(size: 11.5pt, weight: "semibold")[
        #if it.numbering == none [#it.body] else [
          #numbered-heading(counter(heading).display(it.numbering), it.body, number-width: 1.65em)
        ]
      ]
    ]
  ]

  show heading.where(level: 3): it => block(
    width: 100%,
    above: 0.9em,
    below: 0.25em,
  )[
    #align(left)[
      #text(size: 10.5pt, weight: "semibold")[
        #if it.numbering == none [#it.body] else [
          #numbered-heading(counter(heading).display(it.numbering), it.body, number-width: 2.1em)
        ]
      ]
    ]
  ]

  body
}
