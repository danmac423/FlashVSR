#figure(
  kind: table,
  caption: [Konfiguracja referencyjna bez kafelkowania przestrzennego na obu platformach sprzętowych],
  [
    #set text(size: 8.5pt)
    #show table.cell.where(y: 0): strong

    #table(
      columns: (auto, auto, auto, auto, auto, auto),
      align: (right, left, right, right, right, right),
      stroke: 0.5pt + luma(150),
      fill: (col, row) => if row == 0 { luma(230) } else { none },
      table.header([Wejście], [RTX 3080], [Wolne], [Nieudana alokacja], [A100 — szczyt \[MiB\]], [A100 — czas \[s\]]),

      [192 $times$ 352], [przepełnienie], [119 MiB], [434 MiB], [14 844], [9,21],
      [256 $times$ 448], [przepełnienie], [412 MiB], [736 MiB], [23 512], [13,92],
    )],
) <tab:referencja>
