#figure(
  kind: table,
  caption: [Metryki jakości dla różnych rozmiarów kafla, zbiór YouHQ40. Pozostałe parametry jak w konfiguracji z kafelkowaniem z @tab:jakosc-youhq40],
  [
    #set text(size: 8.5pt)
    #show table.cell.where(y: 0): strong

    #table(
      columns: (auto, auto, auto, auto, auto, auto, auto, auto),
      align: (left, right, right, right, right, right, right, right),
      stroke: 0.5pt + luma(150),
      fill: (col, row) => if row == 0 { luma(230) } else { none },
      table.header([Konfiguracja], [PSNR], [SSIM], [LPIPS], [NIQE], [MUSIQ], [CLIPIQA], [DOVER]),

      [kafelkowanie 192], [21,54], [0,599], [0,410], [4,149], [59,42], [0,443], [11,78],
      [kafelkowanie 160], [21,47], [0,594], [0,415], [4,092], [60,46], [0,468], [12,18],
      [kafelkowanie 128], [21,49], [0,592], [0,427], [4,122], [59,78], [0,474], [11,17],
    )],
) <tab:jakosc-youhq40-kafle>
