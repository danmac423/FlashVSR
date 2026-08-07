#figure(
  kind: table,
  caption: [Zmiana metryk względem szczebla poprzedniego, zbiór VideoLQ],
  [
    #set text(size: 8.5pt)
    #show table.cell.where(y: 0): strong

    #table(
      columns: (auto, auto, auto, auto, auto),
      align: (left, right, right, right, right),
      stroke: 0.5pt + luma(150),
      fill: (col, row) => if row == 0 { luma(230) } else { none },
      table.header([Konfiguracja], [NIQE], [MUSIQ], [CLIPIQA], [DOVER]),

      [+ kafelkowanie 192], [−0,067], [−1,09], [−0,011], [−0,16],
      [+ SageAttention], [−0,000], [+0,01], [+0,000], [+0,05],
      [+ SpargeAttention], [−0,004], [−0,04], [+0,002], [−0,04],
      [+ INT8 wag i aktywacji], [+0,015], [−0,01], [−0,001], [−0,02],
    )],
) <tab:jakosc-videolq-delty>
