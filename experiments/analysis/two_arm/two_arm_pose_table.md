# Two-arm pose benchmark

Support fractions: `0.5,0.6`

All numerical statistics use successful runs only. Failure counts and success rates use all recorded runs.

| Strategy | Success | Runtime median [s] | Backward search [%] | RAI validation [%] | Other [%] | Replans median | Support moves median | Support steps median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fast reduce support | 9/10 (90%) | 563.09 | 0.95 | 98.97 | 0.08 | 24.0 | 6.0 | 18.0 |
| baseline | 1/10 (10%) | 506.95 | 0.48 | 99.52 | 0.00 | 21.0 | 7.0 | 29.0 |
| fewest mandatory supports | 2/10 (20%) | 691.79 | 0.37 | 99.62 | 0.00 | 23.0 | 8.5 | 28.5 |
| highest first | 3/10 (30%) | 729.67 | 0.79 | 99.21 | 0.01 | 33.0 | 9.0 | 28.0 |
| default | 1/10 (10%) | 1013.59 | 0.32 | 99.67 | 0.00 | 42.0 | 9.0 | 25.0 |
