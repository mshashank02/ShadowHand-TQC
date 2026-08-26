# Native-rigid CoACD MuJoCo/Warp 3.12 validation

Conclusion: **PASS_36**.

This is a new bare-decomposed-surface collision definition. Historical rigid-flex results were not pooled.

| pieces | contract | fixtures | N=500 | 20-substep | geometry p95 / p99 / max (mm) | volume error |
|---:|:---:|:---:|:---:|:---:|---:|---:|
| 36 | PASS | PASS | PASS | PASS | 0.316 / 0.538 / 0.799 | 0.443% |
| 87 | PASS | PASS | PASS | FAIL | 0.239 / 0.478 / 0.799 | 0.324% |

Recommendation: **36 pieces (only passing representation)**.

GPU throughput and memory were benchmarked for the parity-passing 36-piece representation(s) only. See `gpu_benchmarks.csv` for every batch size.

Any selected passing representation starts a new experiment: all 24 objects and every sensor configuration must be regenerated/validated as applicable and retrained. It is not compatible with historical rigid-flex training results.

The production default was not changed, and full RL training was not run.

MuJoCo 3.12 MULTICCD and NATIVECCD were disabled identically on CPU and Warp because Warp 3.12 cannot transfer the unchanged hand model's non-zero-margin self-collision pairs with MULTICCD enabled. Per-geom object contact parameters were unchanged.
