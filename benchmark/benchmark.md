## benchmark


x -> smaples, y -> populations



### v1.0


replace = True, multinomial distribution


|numcpp/numpy|2|4|8|16|32|64|256|1024|4096|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|10|1.44 µs ± 4.03 ns/34.9 µs ± 281 ns|1.53 µs ± 2.18 ns/35.1 µs ± 209 ns|1.68 µs ± 6.78 ns/35.3 µs ± 672 ns|
|100|2.51 µs ± 9.58 ns/35.9 µs ± 193 ns|2.61 µs ± 12.3 ns/35.9 µs ± 440 ns|2.82 µs ± 13.4 ns/36.1 µs ± 214 ns|3.24 µs ± 11.3 ns/36.8 µs ± 606 ns|4.1 µs ± 34.8 ns/37.3 µs ± 301 ns|5.79 µs ± 8.6 ns/39.7 µs ± 666 ns|
|1000|12.3 µs ± 26.4 ns/46.8 µs ± 269 ns|12.4 µs ± 15.5 ns/46.5 µs ± 618 ns|12.7 µs ± 28.1 ns/47.4 µs ± 566 ns|13.3 µs ± 29.2 ns/47.9 µs ± 547 ns|14.6 µs ± 25.1 ns/49 µs ± 347 ns|17 µs ± 28.5 ns/51.8 µs ± 590 ns|31.9 µs ± 51.9 ns/67.1 µs ± 361 ns|
|10000|109 µs ± 221 ns/130 µs ± 581 ns|110 µs ± 180 ns/130 µs ± 540 ns|110 µs ± 317 ns/131 µs ± 515 ns|111 µs ± 263 ns/132 µs ± 468 ns|112 µs ± 209 ns/133 µs ± 243 ns|116 µs ± 591 ns/137 µs ± 625 ns|135 µs ± 303 ns/158 µs ± 730 ns|211 µs ± 524 ns/239 µs ± 1.86 µs|511 µs ± 542 ns/551 µs ± 8.72 µs|
|1000000|20.1 ms ± 204 µs/9.04 ms ± 15.6 µs|20 ms ± 25.5 µs/9.04 ms ± 9.34 µs|20 ms ± 18.4 µs/9.04 ms ± 6.44 µs|20 ms ± 44.2 µs/9.04 ms ± 9.76 µs|20 ms ± 83.1 µs/9.05 ms ± 6.8 µs|20 ms ± 83.8 µs/9.07 ms ± 9.75 µs|20.1 ms ± 31.2 µs/9.12 ms ± 12.4 µs|20.4 ms ± 153 µs/9.34 ms ± 18.2 µs|21 ms ± 94.5 µs/10.2 ms ± 11.8 µs|
|100000000|1.9 s ± 1.33 ms/995 ms ± 658 µs|1.91 s ± 8.83 ms/994 ms ± 659 µs|1.91 s ± 2.41 ms/998 ms ± 5.06 ms|1.9 s ± 1.25 ms/995 ms ± 1.47 ms|1.91 s ± 2.85 ms/995 ms ± 1.64 ms|1.91 s ± 1.69 ms/995 ms ± 742 µs|1.91 s ± 4.05 ms/995 ms ± 441 µs|1.9 s ± 1.81 ms/997 ms ± 477 µs|1.91 s ± 1.71 ms/1e+03 ms ± 644 µs|


replace = False, multinomial distribution


|numcpp/numpy|2|4|8|16|32|64|256|1024|4096|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|10|1.45 µs ± 8.48 ns/78.2 µs ± 656 ns|1.5 µs ± 4.27 ns/97.7 µs ± 754 ns|1.63 µs ± 4.34 ns/129 µs ± 1.1 µs|
|100|2.47 µs ± 13.3 ns/73.8 µs ± 885 ns|2.58 µs ± 8.28 ns/75.7 µs ± 351 ns|2.79 µs ± 17 ns/84.3 µs ± 733 ns|3.23 µs ± 13.1 ns/101 µs ± 674 ns|4.02 µs ± 11.8 ns/116 µs ± 713 ns|5.66 µs ± 20.9 ns/155 µs ± 956 ns|
|1000|12.4 µs ± 34.1 ns/84.9 µs ± 356 ns|12.5 µs ± 28.2 ns/85.3 µs ± 403 ns|12.8 µs ± 12.1 ns/87.6 µs ± 837 ns|13.4 µs ± 28.9 ns/92.3 µs ± 714 ns|14.5 µs ± 14.4 ns/107 µs ± 434 ns|16.9 µs ± 39.6 ns/130 µs ± 614 ns|31.3 µs ± 82.7 ns/202 µs ± 2.08 µs|
|10000|110 µs ± 282 ns/176 µs ± 268 ns|110 µs ± 172 ns/177 µs ± 599 ns|110 µs ± 247 ns/178 µs ± 694 ns|111 µs ± 139 ns/180 µs ± 485 ns|113 µs ± 504 ns/187 µs ± 441 ns|116 µs ± 364 ns/208 µs ± 2.24 µs|134 µs ± 174 ns/323 µs ± 5.49 µs|207 µs ± 416 ns/530 µs ± 1.72 µs|494 µs ± 803 ns/1.45 ms ± 23.8 µs|
|1000000|19.8 ms ± 29.5 µs/12.6 ms ± 32 µs|19.8 ms ± 41.6 µs/12.6 ms ± 20.3 µs|19.8 ms ± 23.6 µs/12.6 ms ± 59.9 µs|19.8 ms ± 18.6 µs/12.6 ms ± 19.6 µs|19.8 ms ± 38.9 µs/12.6 ms ± 19.7 µs|19.8 ms ± 47.9 µs/12.6 ms ± 59.1 µs|19.9 ms ± 34.5 µs/12.9 ms ± 92.6 µs|20.1 ms ± 112 µs/15.6 ms ± 383 µs|20.8 ms ± 95.3 µs/19.4 ms ± 27.3 µs|
|100000000|1.91 s ± 2.77 ms/1.27 s ± 2.64 ms|1.92 s ± 10.8 ms/1.27 s ± 1.49 ms|1.9 s ± 1.51 ms/1.27 s ± 1.49 ms|1.91 s ± 2.87 ms/1.27 s ± 1.55 ms|1.9 s ± 2.57 ms/1.27 s ± 2.29 ms|1.91 s ± 3.14 ms/1.27 s ± 3.13 ms|1.91 s ± 3.92 ms/1.27 s ± 2.36 ms|1.91 s ± 5.82 ms/1.27 s ± 1.56 ms|1.91 s ± 3.35 ms/1.43 s ± 244 ms|







replace = True, uniform distribution


|numcpp/numpy|2|4|8|16|32|64|256|1024|4096|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|10|639 ns ± 1.32 ns/17.1 µs ± 124 ns|674 ns ± 1.47 ns/17.1 µs ± 121 ns|760 ns ± 2.1 ns/17.1 µs ± 180 ns|
|100|634 ns ± 2.25 ns/17.2 µs ± 114 ns|680 ns ± 10 ns/17 µs ± 107 ns|758 ns ± 3.48 ns/17.2 µs ± 92.4 ns|932 ns ± 3.65 ns/17.2 µs ± 69.9 ns|1.29 µs ± 6.88 ns/17.5 µs ± 78.6 ns|2.01 µs ± 5.22 ns/17.6 µs ± 111 ns|
|1000|640 ns ± 2.03 ns/17.3 µs ± 127 ns|672 ns ± 0.683 ns/17.3 µs ± 85.7 ns|758 ns ± 1.41 ns/17.3 µs ± 121 ns|933 ns ± 2.54 ns/17.4 µs ± 112 ns|1.29 µs ± 4.01 ns/17.5 µs ± 123 ns|2.01 µs ± 8.97 ns/17.9 µs ± 108 ns|6.43 µs ± 17.4 ns/19.6 µs ± 91.8 ns|
|10000|638 ns ± 2.59 ns/17.3 µs ± 106 ns|675 ns ± 0.946 ns/17.4 µs ± 196 ns|762 ns ± 3.65 ns/17.4 µs ± 93.9 ns|940 ns ± 1.95 ns/17.5 µs ± 105 ns|1.3 µs ± 4.72 ns/18 µs ± 176 ns|2.01 µs ± 6.62 ns/18.5 µs ± 138 ns|6.38 µs ± 9.16 ns/22.3 µs ± 108 ns|23.6 µs ± 127 ns/35.4 µs ± 91 ns|92.2 µs ± 212 ns/80.7 µs ± 493 ns|
|1000000|667 ns ± 4.76 ns/17.3 µs ± 67.7 ns|704 ns ± 4.52 ns/17.4 µs ± 131 ns|787 ns ± 2.84 ns/17.4 µs ± 92.7 ns|974 ns ± 4.09 ns/17.6 µs ± 88.6 ns|1.36 µs ± 3.98 ns/17.7 µs ± 50.8 ns|2.13 µs ± 27.1 ns/18.1 µs ± 94.9 ns|6.85 µs ± 33.4 ns/20.9 µs ± 126 ns|25.3 µs ± 257 ns/27.8 µs ± 413 ns|101 µs ± 1.46 µs/51.4 µs ± 977 ns|
|100000000|718 ns ± 2.15 ns/17.3 µs ± 135 ns|769 ns ± 2.28 ns/17.4 µs ± 114 ns|947 ns ± 2.89 ns/17.5 µs ± 91.6 ns|1.25 µs ± 6.38 ns/17.6 µs ± 158 ns|1.92 µs ± 1.54 ns/18.1 µs ± 128 ns|3.11 µs ± 21 ns/18.9 µs ± 134 ns|10.5 µs ± 21.6 ns/24 µs ± 91.2 ns|39.7 µs ± 108 ns/40.3 µs ± 198 ns|158 µs ± 375 ns/101 µs ± 411 ns|


replace = False, uniform distribution


|numcpp/numpy|2|4|8|16|32|64|256|1024|4096|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|10|639 ns ± 2.69 ns/12.7 µs ± 59.5 ns|676 ns ± 1.67 ns/12.9 µs ± 73.6 ns|783 ns ± 3.86 ns/13 µs ± 106 ns|
|100|639 ns ± 1.79 ns/14.2 µs ± 52.8 ns|676 ns ± 1.89 ns/14.4 µs ± 93.6 ns|768 ns ± 2.89 ns/14.3 µs ± 66.8 ns|968 ns ± 1.9 ns/14.3 µs ± 52.1 ns|1.35 µs ± 3.49 ns/14.4 µs ± 76.3 ns|2.11 µs ± 14.7 ns/14.4 µs ± 91.1 ns|
|1000|663 ns ± 1.95 ns/27.5 µs ± 228 ns|682 ns ± 1.54 ns/27.4 µs ± 55.5 ns|770 ns ± 2.98 ns/28 µs ± 435 ns|978 ns ± 2.23 ns/27.9 µs ± 341 ns|1.36 µs ± 2.15 ns/27.9 µs ± 306 ns|2.14 µs ± 11.7 ns/28.2 µs ± 533 ns|6.86 µs ± 15 ns/28.9 µs ± 380 ns|
|10000|642 ns ± 1.85 ns/162 µs ± 736 ns|684 ns ± 1.32 ns/163 µs ± 838 ns|771 ns ± 1.36 ns/162 µs ± 805 ns|978 ns ± 2.61 ns/162 µs ± 717 ns|1.37 µs ± 8.03 ns/163 µs ± 622 ns|2.17 µs ± 19.6 ns/162 µs ± 606 ns|6.97 µs ± 11.2 ns/163 µs ± 549 ns|26.2 µs ± 85.4 ns/166 µs ± 1.15 µs|101 µs ± 386 ns/171 µs ± 515 ns|
|1000000|673 ns ± 3.56 ns/16.3 ms ± 82.9 µs|734 ns ± 4.3 ns/16.2 ms ± 18.8 µs|808 ns ± 2.48 ns/16.2 ms ± 50.3 µs|1.01 µs ± 3.56 ns/16.3 ms ± 131 µs|1.44 µs ± 4.52 ns/16.3 ms ± 65.7 µs|2.3 µs ± 18.4 ns/16.3 ms ± 76.4 µs|7.42 µs ± 51.8 ns/16.2 ms ± 48.9 µs|29.3 µs ± 1.11 µs/16.2 ms ± 50.4 µs|109 µs ± 431 ns/16.3 ms ± 90.1 µs|
|100000000|746 ns ± 2.42 ns/4.23 s ± 24.9 ms|780 ns ± 3.01 ns/4.23 s ± 33.9 ms|938 ns ± 2.73 ns/4.22 s ± 29.7 ms|1.28 µs ± 5.75 ns/4.23 s ± 34.5 ms|1.98 µs ± 6.8 ns/4.23 s ± 21.6 ms|3.16 µs ± 8.14 ns/4.22 s ± 24.6 ms|10.8 µs ± 33.3 ns/4.23 s ± 28.8 ms|41.9 µs ± 180 ns/4.23 s ± 23.8 ms|164 µs ± 603 ns/4.23 s ± 23.7 ms|



### v1.1


replace = True, multinomial distribution


|numcpp/numpy|2|4|8|16|32|64|256|1024|4096|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|100000000|922 ms ± 2.36 ms/993 ms ± 747 µs|922 ms ± 3.26 ms/993 ms ± 1.46 ms|923 ms ± 2.53 ms/993 ms ± 525 µs|923 ms ± 4.07 ms/993 ms ± 846 µs|923 ms ± 2.28 ms/994 ms ± 764 µs|922 ms ± 2.51 ms/995 ms ± 1.1 ms|923 ms ± 4.18 ms/993 ms ± 1.37 ms|924 ms ± 3.04 ms/995 ms ± 819 µs|927 ms ± 1.67 ms/998 ms ± 902 µs|



replace = False, multinomial distribution


|numcpp/numpy|2|4|8|16|32|64|256|1024|4096|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|100000000|922 ms ± 3.17 ms/1.26 s ± 2.15 ms|924 ms ± 2.51 ms/1.26 s ± 1.38 ms|923 ms ± 2.03 ms/1.26 s ± 2.38 ms|923 ms ± 4.13 ms/1.26 s ± 1.59 ms|922 ms ± 2.87 ms/1.26 s ± 1.33 ms|924 ms ± 2.13 ms/1.26 s ± 1.99 ms|923 ms ± 2.96 ms/1.27 s ± 6.3 ms|925 ms ± 2.95 ms/1.26 s ± 816 µs|929 ms ± 3.59 ms/1.27 s ± 2.1 ms|



### v1.2 omp parallel


serial vs. parallel


`num_total_nodes` = 100000, `num_total_edges` = 18000000, `replace` = True, `num_neighbours` = 16, randomly


|#threads|#nodes=100000|#nodes=10000|#nodes=1000|#nodes=500|#nodes=128|#nodes=64|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|serial|106 ms ± 367 µs|10.6 ms ± 23.3 µs|909 µs ± 2.3 µs|466 µs ± 2.18 µs|119 µs ± 826 ns|57.5 µs ± 101 ns|
|2|101 ms ± 706 µs|10.6 ms ± 16.9 µs|868 µs ± 250 ns|444 µs ± 1.35 µs|114 µs ± 285 ns|56.6 µs ± 1.06 µs|
|4|91.9 ms ± 657 µs|9.44 ms ± 22 µs|775 µs ± 1.05 µs|394 µs ± 122 ns|99.6 µs ± 83.9 ns|49.5 µs ± 21.2 ns|
|8|86.5 ms ± 261 µs|9.07 ms ± 364 µs|738 µs ± 6.27 µs|374 µs ± 456 ns|96.5 µs ± 431 ns|48.4 µs ± 598 ns|
|16|86 ms ± 606 µs|10.1 ms ± 250 µs|970 µs ± 12.2 µs|555 µs ± 10.6 µs|274 µs ± 7.38 µs|180 µs ± 897 ns|



### numcpp vs. dgl

`paper` vertex & `cites` edge from [MAG240M](https://ogb.stanford.edu/kddcup2021/mag240m/)


`#paper` = 121,751,666, `#cites` = 1,297,748,926


`#hops` = 2, `#neighbors` = 10


|#threads|time(sec)|memory used(GB)|
|:---:|:---:|:---:|
|numcpp(8 threads)|569|30|
|dgl(4 threads)|773|55|
|dgl(32 threads)|758|55|