# One-arm pose benchmark

Support fractions: `0.4,0.5,0.6`

All numerical statistics use successful runs only. Failure counts and success rates use all recorded runs.

| Strategy | Success | Runtime mean [s] | Runtime median [s] | RAI total [%] | Backward search [%] | Unclassified [%] | Replans mean | Replans median | Support moves median | Support steps median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fast reduce support | 10/10 (100%) | 58.73 | 11.79 | 96.44 | 3.32 | 0.24 | 8.6 | 1.0 | 5.0 | 19.0 |
| baseline | 6/10 (60%) | 62.24 | 50.48 | 98.42 | 1.45 | 0.13 | 7.2 | 6.0 | 8.5 | 26.0 |
| fewest mandatory supports | 10/10 (100%) | 202.27 | 27.07 | 97.88 | 2.02 | 0.11 | 28.4 | 3.5 | 7.0 | 30.0 |
| highest first | 8/10 (80%) | 650.59 | 694.37 | 94.97 | 4.99 | 0.04 | 104.4 | 112.0 | 9.0 | 26.0 |
| default | 6/10 (60%) | 311.85 | 87.67 | 97.89 | 2.06 | 0.06 | 49.0 | 15.5 | 12.5 | 21.5 |
