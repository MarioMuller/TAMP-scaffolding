# One-arm pose benchmark

Support fractions: `0.4,0.5,0.6`

All numerical statistics use successful runs only. Failure counts and success rates use all recorded runs.

| Strategy | Success | Runtime mean [s] | Runtime median [s] | RAI total [%] | Backward search [%] | Unclassified [%] | Replans mean | Replans median | Support moves median | Support steps median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fast reduce support | 10/10 (100%) | 165.49 | 14.59 | 95.77 | 4.03 | 0.19 | 27.5 | 0.5 | 5.0 | 19.0 |
| baseline | 9/10 (90%) | 243.15 | 105.02 | 97.09 | 2.84 | 0.07 | 38.0 | 16.0 | 8.0 | 25.0 |
| fewest mandatory supports | 6/10 (60%) | 85.20 | 22.03 | 98.46 | 1.42 | 0.12 | 14.5 | 1.5 | 7.0 | 30.0 |
| highest first | 8/10 (80%) | 257.10 | 83.31 | 97.25 | 2.67 | 0.08 | 40.2 | 13.0 | 9.5 | 25.5 |
| default | 7/10 (70%) | 282.49 | 237.25 | 97.91 | 2.04 | 0.05 | 47.4 | 46.0 | 12.0 | 22.0 |
