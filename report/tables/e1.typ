#figure(
  kind: table,
  caption: [Czas inferencji i szczytowe zużycie pamięci dla dwunastu kombinacji mechanizmu uwagi i trybu kwantyzacji (wejście 192 $times$ 352, kafel 192, 101 klatek)],
  [
    #set text(size: 8.5pt)
    #show table.cell.where(y: 0): strong

    #table(
      columns: (auto, auto, auto, auto, auto, auto),
      align: (left, left, left, right, right, right),
      stroke: 0.5pt + luma(150),
      fill: (col, row) => if row == 0 { luma(230) } else { none },
      table.header([Jądro gęste], [Selekcja rzadka], [Kwantyzacja], [Czas \[s\]], [s/klatkę], [Szczyt \[MiB\]]),

      [SDPA], [blokowo-rzadka], [brak], [26,9], [0,266], [9 110],
      [SDPA], [blokowo-rzadka], [INT8 wag], [27,9], [0,276], [8 780],
      [SDPA], [blokowo-rzadka], [INT8 wag i akt.], [34,8], [0,345], [8 212],
      [SageAttention], [blokowo-rzadka], [brak], [27,3], [0,271], [9 292],
      [SageAttention], [blokowo-rzadka], [INT8 wag], [28,2], [0,279], [8 764],
      [SageAttention], [blokowo-rzadka], [INT8 wag i akt.], [35,2], [0,348], [8 240],
      [SDPA], [SpargeAttention], [brak], [23,4], [0,232], [9 286],
      [SDPA], [SpargeAttention], [INT8 wag], [24,2], [0,240], [8 766],
      [SDPA], [SpargeAttention], [INT8 wag i akt.], [31,1], [0,308], [8 112],
      [SageAttention], [SpargeAttention], [brak], [23,6], [0,234], [9 294],
      [SageAttention], [SpargeAttention], [INT8 wag], [24,4], [0,241], [8 766],
      [SageAttention], [SpargeAttention], [INT8 wag i akt.], [31,3], [0,310], [8 112],
    )],
) <tab:e1>
