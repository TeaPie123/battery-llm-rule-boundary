# RuleVerifier latency benchmark

The benchmark measures serialized user-message parsing plus deterministic five-rule execution. Disk I/O and model inference are excluded.

| Dataset | Samples | Repeats | Median µs/sample | P95 µs/sample | Median samples/s |
|---|---:|---:|---:|---:|---:|
| balanced | 5310 | 20 | 19.91 | 21.00 | 50223.87 |
| natural | 92194 | 10 | 19.87 | 19.96 | 50321.09 |
| boundary | 71 | 1000 | 19.41 | 20.18 | 51527.43 |

These CPU measurements quantify the additional deterministic verification cost in this implementation. They are environment- and serialization-dependent and must not be presented as a universal deployment latency.
