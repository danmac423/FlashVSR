#figure(
  kind: table,
  caption: [Przesunięcie granicy wykonalności: rozmiar kafla niewykonalny w konfiguracji odniesienia staje się wykonalny po zastosowaniu badanych technik (wejście 256 $times$ 448, 101 klatek)],
  [
    #set text(size: 8.5pt)
    #show table.cell.where(y: 0): strong

    #table(
      columns: (auto, auto, auto, auto, auto),
      align: (left, right, left, right, right),
      stroke: 0.5pt + luma(150),
      fill: (col, row) => if row == 0 { luma(230) } else { none },
      table.header([Konfiguracja], [Kafel], [Wynik], [Czas \[s\]], [Szczyt \[MiB\]]),

      [odniesienia], [192], [wykonalna], [81,6], [9 292],
      [odniesienia], [224], [przepełnienie], [—], [—],
      [SageAttention / SpargeAttention / INT8 wag i akt.], [224], [wykonalna], [132,8], [8 920],
    )],
) <tab:granica>
