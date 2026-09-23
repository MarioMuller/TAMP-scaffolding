# Rigidity Grid Comparison

All calculated metrics use successful runs only. Success reports completed runs out of all attempted runs.

| Removal strategy | Support target | Success | Mean runtime successful runs (s) | Median successful runtime (s) | Mean support moves | Mean support steps | Mean expansions | Mean rigidity checks |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| default | lowest first | 5/5 | 3.17 | 3.18 | 67.80 | 99.80 | 185.6 | 212.4 |
| default | least connected | 4/5 | 3.14 | 3.11 | 70.25 | 97.25 | 159.2 | 187.2 |
| default | random | 3/5 | 3.50 | 3.55 | 64.67 | 108.00 | 176.3 | 209.7 |
| default | furthest from removed | 5/5 | 3.44 | 3.37 | 69.60 | 94.80 | 156.6 | 190.0 |
| highest first | lowest first | 5/5 | 7.46 | 7.42 | 29.20 | 211.20 | 461.8 | 480.0 |
| highest first | least connected | 4/5 | 7.21 | 7.24 | 34.25 | 210.75 | 456.2 | 477.5 |
| highest first | random | 1/5 | 7.26 | 7.26 | 36.00 | 210.00 | 443.0 | 487.0 |
| highest first | furthest from removed | 0/5 | n.a. | n.a. | n.a. | n.a. | n.a. | n.a. |
| fewest mandatory supports | lowest first | 5/5 | 14.49 | 13.78 | 28.80 | 241.40 | 1010.4 | 1023.8 |
| fewest mandatory supports | least connected | 5/5 | 14.08 | 13.77 | 29.60 | 244.40 | 1054.0 | 1068.4 |
| fewest mandatory supports | random | 5/5 | 23.36 | 18.57 | 31.20 | 243.40 | 2021.4 | 2064.4 |
| fewest mandatory supports | furthest from removed | 5/5 | 83.52 | 113.04 | 30.20 | 243.80 | 9156.2 | 9548.2 |
| fast reduce support | lowest first | 5/5 | 28.25 | 27.76 | 26.80 | 128.20 | 172.4 | 1970.8 |
| fast reduce support | least connected | 4/5 | 31.56 | 32.05 | 27.75 | 136.00 | 167.8 | 2113.0 |
| fast reduce support | random | 3/5 | 35.79 | 32.22 | 27.67 | 130.33 | 258.7 | 3009.0 |
| fast reduce support | furthest from removed | 5/5 | 38.28 | 33.57 | 25.80 | 130.00 | 245.2 | 3181.6 |
| baseline | lowest first | 1/5 | 37.07 | 37.07 | 39.00 | 270.00 | 3943.0 | 4116.0 |
| baseline | least connected | 2/5 | 39.33 | 39.33 | 38.50 | 282.50 | 3842.0 | 3881.5 |
| baseline | random | 2/5 | 83.26 | 83.26 | 33.00 | 283.00 | 8943.5 | 9117.0 |
| baseline | furthest from removed | 1/5 | 62.05 | 62.05 | 34.00 | 278.00 | 5978.0 | 6053.0 |
