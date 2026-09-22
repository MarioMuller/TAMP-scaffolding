# Two-arm pose benchmark

Support fractions: `0.5,0.6`

All numerical statistics use successful runs only. Failure counts and success rates use all recorded runs.

| Strategy | Success | Runtime median [s] | Replans median | Support moves median | Support steps median |
| --- | --- | --- | --- | --- | --- |
| fast reduce support | 9/10 (90%) | 563.09 | 24.0 | 6.0 | 18.0 |
| baseline | 1/10 (10%) | 506.95 | 21.0 | 7.0 | 29.0 |
| fewest mandatory supports | 2/10 (20%) | 691.79 | 23.0 | 8.5 | 28.5 |
| highest first | 3/10 (30%) | 729.67 | 33.0 | 9.0 | 28.0 |
| default | 1/10 (10%) | 1013.59 | 42.0 | 9.0 | 25.0 |
