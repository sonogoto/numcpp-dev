## test results


### serial vs. parallel

`num_total_nodes` = 100000, `num_total_edges` = 18000000, `replace` = True, `num_neighbours` = 16

|#threads|#nodes=100000|#nodes=10000|#nodes=1000|#nodes=500|#nodes=128|#nodes=64|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|serial|106 ms ± 367 µs|10.6 ms ± 23.3 µs|909 µs ± 2.3 µs|466 µs ± 2.18 µs|119 µs ± 826 ns|57.5 µs ± 101 ns|
|2|101 ms ± 706 µs|10.6 ms ± 16.9 µs|868 µs ± 250 ns|444 µs ± 1.35 µs|114 µs ± 285 ns|56.6 µs ± 1.06 µs|
|4|91.9 ms ± 657 µs|9.44 ms ± 22 µs|775 µs ± 1.05 µs|394 µs ± 122 ns|99.6 µs ± 83.9 ns|49.5 µs ± 21.2 ns|
|8|86.5 ms ± 261 µs|9.07 ms ± 364 µs|738 µs ± 6.27 µs|374 µs ± 456 ns|96.5 µs ± 431 ns|48.4 µs ± 598 ns|
|16|86 ms ± 606 µs|10.1 ms ± 250 µs|970 µs ± 12.2 µs|555 µs ± 10.6 µs|274 µs ± 7.38 µs|180 µs ± 897 ns|