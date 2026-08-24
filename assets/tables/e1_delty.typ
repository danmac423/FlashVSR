#figure(
  kind: table,
  caption: [Zmiany względem konfiguracji odniesienia],
  [
    #set text(size: 8.5pt)
    #show table.cell.where(y: 0): strong

    #table(
      columns: (auto, auto, auto, auto),
      align: (left, right, right, right),
      stroke: 0.5pt + luma(150),
      fill: (col, row) => if row == 0 { luma(230) } else { none },
      table.header([Konfiguracja], [Czas (wzgl.)], [Pamięć], [Pamięć (wzgl.)]),

      [SDPA / Block Sparse Attention / brak], [0,0%], [0 MiB], [0,0%],
      [SDPA / Block Sparse Attention / INT8 wag], [+3,6%], [-330 MiB], [-3,6%],
      [SDPA / Block Sparse Attention / INT8 wag i akt.], [+29,6%], [-898 MiB], [-9,9%],
      [SageAttention / Block Sparse Attention / brak], [+1,7%], [+182 MiB], [+2,0%],
      [SageAttention / Block Sparse Attention / INT8 wag], [+4,7%], [-346 MiB], [-3,8%],
      [SageAttention / Block Sparse Attention / INT8 wag i akt.], [+30,7%], [-870 MiB], [-9,5%],
      [SDPA / SpargeAttention / brak], [-12,9%], [+176 MiB], [+1,9%],
      [SDPA / SpargeAttention / INT8 wag], [-9,9%], [-344 MiB], [-3,8%],
      [SDPA / SpargeAttention / INT8 wag i akt.], [+15,7%], [-998 MiB], [-11,0%],
      [SageAttention / SpargeAttention / brak], [-12,3%], [+184 MiB], [+2,0%],
      [SageAttention / SpargeAttention / INT8 wag], [-9,4%], [-344 MiB], [-3,8%],
      [SageAttention / SpargeAttention / INT8 wag i akt.], [+16,4%], [-998 MiB], [-11,0%],
    )],
) <tab:e1-delty>
