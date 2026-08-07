#figure(
  kind: table,
  caption: [Metryki bezreferencyjne dla zbioru VideoLQ, wartości uśrednione po klipach],
  [
    #set text(size: 8.5pt)
    #show table.cell.where(y: 0): strong

    #table(
      columns: (auto, auto, auto, auto, auto),
      align: (left, right, right, right, right),
      stroke: 0.5pt + luma(150),
      fill: (col, row) => if row == 0 { luma(230) } else { none },
      table.header([Konfiguracja], [NIQE], [MUSIQ], [CLIPIQA], [DOVER]),

      [bez kafelkowania], [3,937], [52,02], [0,402], [8,02],
      [+ kafelkowanie 192], [3,870], [50,93], [0,391], [7,86],
      [+ SageAttention], [3,869], [50,94], [0,391], [7,91],
      [+ SpargeAttention], [3,865], [50,91], [0,393], [7,87],
      [+ INT8 wag i aktywacji], [3,881], [50,90], [0,392], [7,84],
    )],
) <tab:jakosc-videolq>
