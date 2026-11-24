| tag | prefix cache | guided | batch | median wall (s) | docs/s | output tok/s | total tok/s | parse rate | repeats |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | on | on | 1 | 124.47 | 0.24 | 20 | 545 | 100.0% | 3 |
| base | on | on | 4 | 64.96 | 0.46 | 38 | 1044 | 100.0% | 3 |
| base | on | on | 8 | 53.19 | 0.56 | 47 | 1276 | 100.0% | 3 |
| base | on | on | 32 | 46.20 | 0.65 | 54 | 1469 | 100.0% | 3 |
| base | on | off | 1 | 124.48 | 0.24 | 20 | 545 | 93.3% | 3 |
| base | on | off | 4 | 64.98 | 0.46 | 38 | 1044 | 93.3% | 3 |
| base | on | off | 8 | 53.20 | 0.56 | 46 | 1275 | 93.3% | 3 |
| base | on | off | 32 | 46.21 | 0.65 | 54 | 1468 | 93.3% | 3 |
| gpu095 | on | on | 32 | 43.74 | 0.69 | 57 | 1551 | 100.0% | 3 |
| no_prefix | off | on | 32 | 53.91 | 0.56 | 46 | 1259 | 100.0% | 3 |

- Batching: batch 32 vs batch 1 (guided on) -> 2.69x throughput
- Guided decoding cost at batch 32: throughput +0.0% vs guided off
- Prefix caching: on vs off at batch 32 -> 1.17x throughput
- KV cache [base, max_model_len=16384]: 36,864 tokens
- KV cache [gpu095, max_model_len=16384]: 50,176 tokens
- KV cache [len4096, max_model_len=4096]: 36,864 tokens
- KV cache [no_prefix, max_model_len=16384]: 38,912 tokens
