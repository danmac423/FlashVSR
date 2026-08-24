#figure(
  kind: table,
  caption: flex-caption(
    [Metryki jakości dla zbioru YouHQ40, wartości uśrednione po klipach],
    [Metryki jakości dla zbioru YouHQ40],
  ),
  [
    #set text(size: 8.5pt)
    #show table.cell.where(y: 0): strong

    #table(
      columns: (auto, auto, auto, auto, auto, auto, auto, auto),
      align: (left, right, right, right, right, right, right, right),
      stroke: 0.5pt + luma(150),
      fill: (col, row) => if row == 0 { luma(230) } else { none },
      table.header([Konfiguracja], [PSNR#sym.arrow.t], [SSIM#sym.arrow.t], [LPIPS#sym.arrow.b], [NIQE#sym.arrow.b], [MUSIQ#sym.arrow.t], [CLIPIQA#sym.arrow.t], [DOVER#sym.arrow.t]),

      [bazowa], [21,79], [0,591], [0,393], [4,245], [60,78], [0,486], [12,52],
      [\+ kafelkowanie 192], [21,54], [0,599], [0,410], [4,149], [59,42], [0,443], [11,78],
      [\+ SageAttention], [21,54], [0,599], [0,410], [4,149], [59,41], [0,444], [11,66],
      [\+ SpargeAttention], [21,55], [0,599], [0,411], [4,167], [58,94], [0,442], [11,63],
      [\+ INT8 wag i aktywacji], [21,54], [0,599], [0,410], [4,157], [58,97], [0,440], [11,60],
    )],
) <tab:jakosc-youhq40>
