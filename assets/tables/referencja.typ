#figure(
  kind: table,
  caption: flex-caption(
    [Zapotrzebowanie konfiguracji bazowej, zmierzone na akceleratorze A100],
    [Zapotrzebowanie konfiguracji bazowej],
  ),
  [
    #set text(size: 8.5pt)
    #show table.cell.where(y: 0): strong

    #table(
      columns: (auto, auto, auto, auto),
      align: (right, right, right, right),
      stroke: 0.5pt + luma(150),
      fill: (col, row) => if row == 0 { luma(230) } else { none },
      table.header([Wejście], [Szczyt \[MiB\]], [Czas \[s\]], [Krotność pojemności RTX 3080]),

      [192 $times$ 352], [14 844], [9,21], [1,50$times$],
      [256 $times$ 448], [23 512], [13,92], [2,38$times$],
    )],
) <tab:referencja>
