# 20-Rod Support Search Comparison

Only successful runs are included.

| Method | Successful runs | Distinct seeds | Unique removal paths | Support steps min-max | Support moves min-max | Peak supports min-max | Runtime [s] min-max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fast reduce support | 10 | 10 | 8 | 19-19 | 5-5 | 2-2 | 0.189-0.201 |
| Random removal + random supports | 10 | 10 | 10 | 21-31 | 5-9 | 2-2 | 0.082-0.163 |
| Reduced overall support steps | 1 | 1 | 1 | 11-11 | 6-6 | 1-1 | 850.557-850.557 |
| Reduced overall supports | 1 | 1 | 1 | 19-19 | 4-4 | 2-2 | 839.766-839.766 |

Fast-reduction seeds: `200452,210452,220452,230452,240452,250452,260452,270452,280452,290452`.

The reference values are the best results in the supplied expensive-search files. They are exact for the transition choices explored by those searches, but they do not prove a global physical optimum over every possible support-target assignment.
