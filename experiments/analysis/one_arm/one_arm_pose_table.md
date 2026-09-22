# One-arm pose benchmark

Support fractions: `0.5,0.6`

All numerical statistics use successful runs only. Failure counts and success rates use all recorded runs.

| Strategy | Success | Runtime median [s] | Backward search [%] | RAI validation [%] | Other [%] | Replans median | Support moves median | Support steps median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fast reduce support | 9/10 (90%) | 6.90 | 3.74 | 96.24 | 0.02 | 0.0 | 5.0 | 19.0 |
| baseline | 6/10 (60%) | 235.67 | 0.66 | 99.34 | 0.01 | 10.0 | 8.5 | 28.0 |
| fewest mandatory supports | 5/10 (50%) | 49.20 | 1.00 | 98.99 | 0.01 | 9.0 | 7.0 | 30.0 |
| highest first | 5/10 (50%) | 266.35 | 1.12 | 98.87 | 0.01 | 18.0 | 9.0 | 25.0 |
| default | 3/10 (30%) | 133.94 | 1.01 | 98.98 | 0.01 | 6.0 | 13.0 | 22.0 |
