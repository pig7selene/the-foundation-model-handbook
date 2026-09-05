#import "typography.typ": setup-chapter-typography

#let abstract(body) = align(center)[
  #block(
    width: 124mm,
    above: 0pt,
    below: 1.55em,
  )[
    #set par(first-line-indent: 0pt, leading: 0.63em, spacing: 0.7em)
    #align(center)[#text(size: 10pt, weight: "semibold")[Abstract]]
    #v(0.75em)
    #align(left)[#text(size: 9.4pt)[#body]]
  ]
]

#let chapter-bibliography(path) = {
  set text(size: 8.6pt)
  set par(first-line-indent: 0pt, leading: 0.52em)
  bibliography(path, title: none)
}

#let academic-table(
  columns: auto,
  header: (),
  rows: (),
  align: auto,
  inset: (x: 5pt, y: 4pt),
) = table(
  columns: columns,
  align: align,
  inset: inset,
  stroke: none,
  table.hline(y: 0, stroke: 0.75pt + black),
  table.header(..header),
  table.hline(y: 1, stroke: 0.45pt + black),
  ..rows,
  table.hline(stroke: 0.75pt + black),
)

#let technical-chapter(
  title: content,
  body,
) = {
  setup-chapter-typography({
    set document(title: title, author: ("pig7selene",))
    set page(
      paper: "a4",
      margin: (top: 24mm, bottom: 25mm, left: 31mm, right: 27mm),
      header: none,
      footer: context {
        if counter(page).get().first() == 1 {
          []
        } else {
          align(center)[#text(size: 8.7pt)[#counter(page).display("1")]]
        }
      },
    )

    align(center)[
      #text(size: 22pt, weight: "semibold")[#title]
      #v(1.65em)
      #text(size: 9.5pt)[pig7selene]
      #v(0.55em)
      #text(size: 8.9pt, fill: luma(55%))[September 5, 2026]
      #v(2.15em)
    ]

    body
  })
}
