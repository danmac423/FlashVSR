#figure(
  kind: table,
  caption: flex-caption(
    [Czas inferencji i szczytowe zużycie pamięci w funkcji rozmiaru kafla (wejście 256 $times$ 448, zakładka 24)],
    [Czas inferencji i szczytowe zużycie pamięci w funkcji rozmiaru kafla],
  ),
  [
    #set text(size: 8.5pt)
    #show table.cell.where(y: 0): strong

    #table(
      columns: (auto, auto, auto, auto, auto, auto, auto),
      align: (right, right, right, right, right, right, right),
      stroke: 0.5pt + luma(150),
      fill: (col, row) => if row == 0 { luma(230) } else { none },
      table.header([Rozmiar kafla], [Liczba kafli], [Redundancja], [Czas \[s\]], [s/klatkę], [Czas/kafel \[s\]], [Szczyt \[MiB\]]),

      [128], [15], [2,14], [81,32], [0,805], [5,42], [6 472],
      [160], [8], [1,79], [72,78], [0,721], [9,10], [7 386],
      [192], [6], [1,93], [81,62], [0,808], [13,60], [9 292],
      [224], [6], [2,62], [przepełnienie], [—], [—], [—],
      [256], [2], [1,14], [przepełnienie], [—], [—], [—],
    )],
) <tab:e2>
