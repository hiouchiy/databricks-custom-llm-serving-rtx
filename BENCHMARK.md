# Benchmark Results

## Environment

| Item | Value |
|---|---|
| GPU | RTX PRO 6000 (96 GB GDDR7) — `GPU_LARGE_RTX` |
| Tensor Parallel | 1 (single GPU) |
| vLLM | 0.19.1 |
| `--max-model-len` | 8192 |
| Prompt | verbose essay prompt (~40 tokens) |
| `max_tokens` | 256 |

---

## FP8 — Qwen3.5-27B-FP8

| Item | Value |
|---|---|
| Weights | ~28.5 GiB loaded |
| `--max-num-seqs` | 128 |
| KV cache budget | ~69 GB |

### Concurrency sweep

| C | sys_tps (tok/s) | p50 latency | TTFT p50 | TPOT p50 | eff_decode_C |
|---|---|---|---|---|---|
| 1 | 37.9 | 6.76s | 210ms | 25.7ms | 1.0 |
| 2 | 76.7 | 6.65s | 245ms | 25.2ms | 2.0 |
| 4 | 148.3 | 6.87s | 435ms | 25.3ms | 3.96 |
| 8 | 309.8 | 6.59s | 254ms | 24.9ms | 7.92 |
| 16 | 544.9 | 7.49s | 732ms | 26.5ms | 14.97 |
| 32 | 1,016 | 8.02s | 347ms | 30.1ms | 31.45 |
| 48 | 1,342 | 9.12s | 503ms | 33.8ms | 46.83 |
| 64 | 1,631 | 10.03s | 482ms | 37.5ms | 62.50 |
| 80 | 1,811 | 11.29s | 704ms | 41.5ms | 77.10 |
| 96 | 2,001 | 12.26s | 755ms | 45.1ms | 92.47 |
| 112 | 2,164 | 13.20s | 743ms | 48.8ms | 107.79 |
| **128** | **2,282** | **14.33s** | 907ms | 52.6ms | 123.23 |

---

## BF16 — Qwen3.5-27B

| Item | Value |
|---|---|
| Weights | ~54 GiB |
| `--max-num-seqs` | 64 |
| KV cache budget | ~30 GB |

### Concurrency sweep

| C | sys_tps (tok/s) | p50 latency | TTFT p50 | TPOT p50 | eff_decode_C |
|---|---|---|---|---|---|
| 1 | 26.6 | 9.63s | 178ms | 37.1ms | 1.0 |
| 2 | 48.0 | 10.66s | 406ms | 40.2ms | 2.0 |
| 4 | 96.0 | 10.63s | 276ms | 40.6ms | 4.0 |
| 8 | 184.0 | 11.09s | 655ms | 40.9ms | 7.7 |
| 16 | 357.1 | 11.43s | 350ms | 43.5ms | 15.8 |
| 32 | 643.6 | 12.69s | 386ms | 48.2ms | 31.5 |
| 48 | 867.6 | 14.12s | 587ms | 53.1ms | 46.8 |
| **64** | **1,076** | **15.18s** | 659ms | 57.0ms | 62.2 |

---

## FP8 vs BF16 comparison

| Metric | FP8 (C=1) | BF16 (C=1) | FP8 (C=32) | BF16 (C=32) |
|---|---|---|---|---|
| sys_tps | 37.9 | 26.6 | 1,016 | 644 |
| TPOT p50 | 25.7ms | 37.1ms | 30.1ms | 48.2ms |
| p50 latency | 6.76s | 9.63s | 8.02s | 12.69s |
| Peak tps | 2,282 (C=128) | 1,076 (C=64) | — | — |

**FP8 throughput advantage:** 1.4–2.1× across concurrency levels.
**Continuous batching:** fully effective on both variants — `eff_decode_C ≈ C` throughout.
**Saturation:** FP8 shows diminishing returns around C=64–80 (TPOT grows from 25ms to 38ms);
BF16 shows similar pattern around C=32–48.
