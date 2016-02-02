#!/bin/bash

# data
./acor_data.py -c 5 
# sample
./acor_sample.py -c 5 -n 1000000
# quadrature
./acor_test.py -c 5 -k 0 --quad -l -2 -u 2
./acor_test.py -c 5 -k 1 --quad -l  2 -u 7
./acor_test.py -c 5 -k 2 --quad -l  0 -u 7
./acor_test.py -c 5 -k 3 --quad -l  4 -u 9
 # plot
./acor_plot.py -c 5 -k 0 -l -2 -u 2 --theory
./acor_plot.py -c 5 -k 1 -l  2 -u 7 --theory
./acor_plot.py -c 5 -k 2 -l  0 -u 7 --theory
./acor_plot.py -c 5 -k 3 -l  4 -u 9 --theory
./acor_plot.py -c 5 --two