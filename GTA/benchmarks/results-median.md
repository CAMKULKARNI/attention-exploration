### Median Time To First Token (TTFT) in Seconds ###
| exp_name                           |    512 |   1024 |   2048 |   4096 |
|:-----------------------------------|-------:|-------:|-------:|-------:|
| Exp 1: Vanilla MHA (24 Layers)     | 0.0382 | 0.0755 | 0.1874 | 0.5737 |
| Exp 2: GQA + RoPE (24 Layers)      | 0.0325 | 0.1769 | 0.1794 | 0.4736 |
| Exp 3: TA + ALiBi (24 Layers)     | 0.0345 | 0.0725 | 0.1877 | 0.5097 |
| Exp 4: GQA + RoPE (28 Layers)      | 0.0419 | 0.1152 | 0.2142 | 0.6131 |
| Exp 5: GQA + RoPE (30 Layers)      | 0.0683 | 0.1619 | 0.3188 | 0.5661 |
| Exp 6: TA + ALiBi (28 Layers)     | 0.0362 | 0.0815 | 0.2147 | 0.5796 |
| Exp 7: TA + ALiBi (30 Layers)     | 0.0405 | 0.0901 | 0.2215 | 0.6194 |
| Exp 8: MHA Iso-KV-Cache (6 Layers) | 0.0128 | 0.025  | 0.0597 | 0.1546 |
| Exp 9: MHA + RoPE (24 Layers)      | 0.0395 | 0.0798 | 0.187  | 0.4957 |
| Exp 10: MHA + RoPE (28 Layers)     | 0.0426 | 0.0893 | 0.2139 | 0.5549 |
| Exp 11: MHA + RoPE (30 Layers)     | 0.0476 | 0.0954 | 0.2259 | 0.5886 |
| Exp 12: GTA + ALiBi (24 Layers)   | 0.0313 | 0.1587 | 0.182  | 0.4925 |
| Exp 13: GTA + ALiBi (28 Layers)   | 0.0426 | 0.1857 | 0.2168 | 0.5705 |
| Exp 14: GTA + ALiBi (30 Layers)   | 0.0474 | 0.1578 | 0.2588 | 0.5957 |

### Median Time Per Output Token (TPOT) in Seconds ###
| exp_name                           |    512 |   1024 |   2048 |   4096 |
|:-----------------------------------|-------:|-------:|-------:|-------:|
| Exp 1: Vanilla MHA (24 Layers)     | 0.0066 | 0.0075 | 0.0078 | 0.0089 |
| Exp 2: GQA + RoPE (24 Layers)      | 0.0065 | 0.007  | 0.0069 | 0.0082 |
| Exp 3: TA + ALiBi (24 Layers)     | 0.006  | 0.0062 | 0.007  | 0.0081 |
| Exp 4: GQA + RoPE (28 Layers)      | 0.0074 | 0.0083 | 0.0085 | 0.009  |
| Exp 5: GQA + RoPE (30 Layers)      | 0.0084 | 0.0085 | 0.0088 | 0.0098 |
| Exp 6: TA + ALiBi (28 Layers)     | 0.0063 | 0.0064 | 0.0075 | 0.0095 |
| Exp 7: TA + ALiBi (30 Layers)     | 0.0064 | 0.0081 | 0.0079 | 0.0094 |
| Exp 8: MHA Iso-KV-Cache (6 Layers) | 0.0028 | 0.0033 | 0.0037 | 0.0038 |
| Exp 9: MHA + RoPE (24 Layers)      | 0.0077 | 0.0082 | 0.008  | 0.0092 |
| Exp 10: MHA + RoPE (28 Layers)     | 0.0084 | 0.0083 | 0.0093 | 0.0107 |
| Exp 11: MHA + RoPE (30 Layers)     | 0.0084 | 0.0095 | 0.0101 | 0.0114 |
| Exp 12: GTA + ALiBi (24 Layers)   | 0.0059 | 0.0059 | 0.006  | 0.0067 |
| Exp 13: GTA + ALiBi (28 Layers)   | 0.0062 | 0.0064 | 0.0068 | 0.0072 |
| Exp 14: GTA + ALiBi (30 Layers)   | 0.0068 | 0.0075 | 0.0079 | 0.0077 |

### Median Peak Incremental VRAM Usage in MB ###
| exp_name                           |     512 |    1024 |    2048 |    4096 |
|:-----------------------------------|--------:|--------:|--------:|--------:|
| Exp 1: Vanilla MHA (24 Layers)     | 530.671 | 630.919 | 1046.12 | 2563.04 |
| Exp 2: GQA + RoPE (24 Layers)      | 750.284 | 738.919 | 1116.88 | 2166.16 |
| Exp 3: TA + ALiBi (24 Layers)     | 676.443 | 702.919 | 1159.81 | 2400.31 |
| Exp 4: GQA + RoPE (28 Layers)      | 637.441 | 805.757 | 1148.56 | 2124.55 |
| Exp 5: GQA + RoPE (30 Layers)      | 680.759 | 805.757 | 1143.56 | 2138.05 |
| Exp 6: TA + ALiBi (28 Layers)     | 612.759 | 724.757 | 1160.13 | 2392.05 |
| Exp 7: TA + ALiBi (30 Layers)     | 642.601 | 790.757 | 1076.39 | 2423.89 |
| Exp 8: MHA Iso-KV-Cache (6 Layers) | 819.968 | 794.966 | 1154.76 | 2102.29 |
| Exp 9: MHA + RoPE (24 Layers)      | 528.671 | 630.919 | 1070.54 | 2954.04 |
| Exp 10: MHA + RoPE (28 Layers)     | 546.599 | 627.757 | 1148.38 | 3039.31 |
| Exp 11: MHA + RoPE (30 Layers)     | 536.671 | 653.591 | 1188.17 | 3123.32 |
| Exp 12: GTA + ALiBi (24 Layers)   | 790.284 | 823.283 | 1147.97 | 2106.91 |
| Exp 13: GTA + ALiBi (28 Layers)   | 668.441 | 853.757 | 1223.56 | 2114.32 |
| Exp 14: GTA + ALiBi (30 Layers)   | 694.759 | 850.007 | 1218.56 | 2120.57 |

### Model Parameter Count ###
| exp_name                           | params   |
|:-----------------------------------|:---------|
| Exp 1: Vanilla MHA (24 Layers)     | 354.6M   |
| Exp 2: GQA + RoPE (24 Layers)      | 315.8M   |
| Exp 3: TA + ALiBi (24 Layers)     | 328.39M  |
| Exp 4: GQA + RoPE (28 Layers)      | 359.86M  |
| Exp 5: GQA + RoPE (30 Layers)      | 381.89M  |
| Exp 6: TA + ALiBi (28 Layers)     | 374.54M  |
| Exp 7: TA + ALiBi (30 Layers)     | 397.62M  |
| Exp 8: MHA Iso-KV-Cache (6 Layers) | 128.04M  |
| Exp 9: MHA + RoPE (24 Layers)      | 353.55M  |
| Exp 10: MHA + RoPE (28 Layers)     | 403.9M   |
| Exp 11: MHA + RoPE (30 Layers)     | 429.08M  |
| Exp 12: GTA + ALiBi (24 Layers)   | 309.51M  |
| Exp 13: GTA + ALiBi (28 Layers)   | 352.52M  |
| Exp 14: GTA + ALiBi (30 Layers)   | 374.03M  |