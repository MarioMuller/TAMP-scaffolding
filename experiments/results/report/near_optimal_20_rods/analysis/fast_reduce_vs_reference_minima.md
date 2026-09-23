# 20-Rod Support Search Comparison

Only successful runs are included.

| Method | Successful runs | Support steps (range / avg. / median) | Support moves (range / avg. / median) | Runtime [s] (range / avg. / median) |
|---|---:|---:|---:|---:|
| Uninformed baseline + random supports | 10 | 21.0-31.0 / 26.1 / 25.5 | 5.0-9.0 / 7.0 / 7.0 | 0.085-0.186 / 0.116 / 0.110 |
| Uninformed baseline + lowest | 10 | 25.0-31.0 / 28.5 / 29.5 | 7.0-12.0 / 8.6 / 8.5 | 0.097-0.226 / 0.132 / 0.119 |
| Default + lowest | 10 | 21.0-21.0 / 21.0 / 21.0 | 14.0-14.0 / 14.0 / 14.0 | 0.098-0.102 / 0.100 / 0.099 |
| Default + furthest | 10 | 20.0-24.0 / 21.2 / 20.0 | 11.0-15.0 / 13.8 / 15.0 | 0.110-0.119 / 0.115 / 0.114 |
| Fast reduce support + lowest | 10 | 19.0-19.0 / 19.0 / 19.0 | 5.0-5.0 / 5.0 / 5.0 | 0.299-0.368 / 0.332 / 0.336 |
| Fast reduce support + furthest | 10 | 19.0-19.0 / 19.0 / 19.0 | 5.0-6.0 / 5.5 / 5.5 | 0.284-0.314 / 0.295 / 0.294 |
| Reduced overall support steps | 1 | 11.0 | 6.0 | 850.557 |
| Reduced overall supports | 1 | 19.0 | 4.0 | 839.766 |

Cells report range / average / median over successful runs.

Heuristic-run seeds: `200452,210452,220452,230452,240452,250452,260452,270452,280452,290452`.

The reference values are the best results in the supplied expensive-search files. They are exact for the transition choices explored by those searches, but they do not prove a global physical optimum over every possible support-target assignment.
