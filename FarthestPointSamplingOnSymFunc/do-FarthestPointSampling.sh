#!/bin/bash

#cleaning the Symmetry function data file for processing
awk '{if (NF == 43) print $0}' function.data > function.cleaned.data 
wait

#extracting the energy data from the input.data file
grep 'energy' input.data | awk '{print $2}' > energy.data
wait

# This script is used to sample the data points from the function.data file using FPS algorithm

echo "Sampling the data points using FPS algorithm"
python3 FPS-N2P2.py -o output.data -n 10000    # -n 10000 is the number of points to be sampled using FPS
wait

rm -rf function.cleaned.data
echo "Done!"