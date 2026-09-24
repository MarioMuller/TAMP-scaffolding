# Rigidity Grid Comparison

All calculated metrics use successful runs only. Success reports completed runs out of all attempted runs.

| Removal strategy | Support target | Success | Mean runtime successful runs (s) | Median successful runtime (s) | Mean support moves | Mean support steps | Mean expansions | Mean rigidity checks |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| default | lowest first | 10/10 | 3.17 | 3.17 | 67.90 | 99.90 | 185.7 | 212.6 |
| default | least connected | 9/10 | 3.10 | 3.08 | 70.56 | 96.22 | 158.4 | 186.3 |
| default | random | 6/10 | 3.85 | 3.56 | 62.50 | 113.17 | 188.3 | 223.2 |
| default | furthest from removed | 10/10 | 3.39 | 3.32 | 70.50 | 93.90 | 154.8 | 187.7 |
| highest first | lowest first | 10/10 | 14.90 | 9.64 | 28.90 | 211.20 | 456.7 | 474.4 |
| highest first | least connected | 7/10 | 10.00 | 8.12 | 33.86 | 210.29 | 445.3 | 467.1 |
| highest first | random | 3/10 | 10.55 | 11.46 | 35.67 | 210.00 | 426.0 | 460.7 |
| highest first | furthest from removed | 1/10 | 6.38 | 6.38 | 30.00 | 210.00 | 365.0 | 388.0 |
| fewest mandatory supports | lowest first | 10/10 | 14.50 | 14.45 | 28.70 | 241.50 | 1009.9 | 1023.8 |
| fewest mandatory supports | least connected | 10/10 | 14.62 | 14.82 | 29.40 | 243.40 | 1072.8 | 1088.0 |
| fewest mandatory supports | random | 8/10 | 22.12 | 18.41 | 31.12 | 242.25 | 1946.5 | 1991.5 |
| fewest mandatory supports | furthest from removed | 10/10 | 81.28 | 107.10 | 30.40 | 244.40 | 8815.6 | 9193.7 |
| fast reduce support | lowest first | 10/10 | 29.13 | 27.71 | 27.00 | 124.60 | 168.1 | 1967.3 |
| fast reduce support | least connected | 9/10 | 30.33 | 31.26 | 28.11 | 125.22 | 170.7 | 2102.7 |
| fast reduce support | random | 7/10 | 31.63 | 29.93 | 27.14 | 121.43 | 204.0 | 2395.0 |
| fast reduce support | furthest from removed | 10/10 | 37.13 | 31.93 | 25.90 | 126.70 | 231.1 | 3075.9 |
| baseline | lowest first | 1/10 | 37.07 | 37.07 | 39.00 | 270.00 | 3943.0 | 4116.0 |
| baseline | least connected | 2/10 | 39.33 | 39.33 | 38.50 | 282.50 | 3842.0 | 3881.5 |
| baseline | random | 4/10 | 53.16 | 55.07 | 35.00 | 279.25 | 5219.2 | 5323.5 |
| baseline | furthest from removed | 1/10 | 62.05 | 62.05 | 34.00 | 278.00 | 5978.0 | 6053.0 |
