#import "typography.typ": setup-typography

#let handbook(title: content, subtitle: content, author: content, body) = {
  setup-typography()

  set document(title: title, author: author)
  set page(
    paper: "a4",
    margin: (top: 28mm, bottom: 27mm, left: 30mm, right: 24mm),
    header: align(center)[#text(size: 8pt, fill: gray)[The Foundation Model Handbook]],
    footer: align(center)[#text(size: 9pt)[#counter(page).display("1")]],
  )

  align(center + horizon)[
    #text(size: 29pt, weight: "semibold")[#title]
    #v(1.4em)
    #text(size: 14pt, style: "italic")[#subtitle]
    #v(3.4em)
    #text(size: 11pt)[#author]
    #v(1.2em)
    #text(size: 10pt)[Foundation Model Study Series]
  ]

  pagebreak()
  body
}
