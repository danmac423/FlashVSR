#figure(
  kind: table,
  caption: [Zmiana metryk względem konfiguracji poprzedniej, zbiór YouHQ40],
  [
    #set text(size: 8.5pt)
    #show table.cell.where(y: 0): strong

    #table(
      columns: (auto, auto, auto, auto, auto, auto, auto, auto),
      align: (left, right, right, right, right, right, right, right),
      stroke: 0.5pt + luma(150),
      fill: (col, row) => if row == 0 { luma(230) } else { none },
      table.header([Konfiguracja], [PSNR#sym.arrow.t], [SSIM#sym.arrow.t], [LPIPS#sym.arrow.b], [NIQE#sym.arrow.b], [MUSIQ#sym.arrow.t], [CLIPIQA#sym.arrow.t], [DOVER#sym.arrow.t]),

      [\+ kafelkowanie 192], [-0,25], [+0,008], [+0,016], [-0,096], [-1,36], [-0,043], [-0,73],
      [\+ SageAttention], [0,00], [0,000], [0,000], [0,000], [-0,01], [0,000], [-0,12],
      [\+ SpargeAttention], [+0,01], [0,000], [+0,001], [+0,018], [-0,47], [-0,001], [-0,03],
      [\+ INT8 wag i aktywacji], [-0,01], [0,000], [0,000], [-0,009], [+0,03], [-0,002], [-0,03],
    )],
) <tab:jakosc-youhq40-delty>
