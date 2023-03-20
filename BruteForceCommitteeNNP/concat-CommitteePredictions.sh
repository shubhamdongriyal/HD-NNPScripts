#!/bin/bash

# this script creates a concatenated files of predictions of energies for each structure by committee of neural networks

echo 'Concat predictions from all the committee neural network potential predictions'

paste NN*/forward_pass.txt | cut -f 1,2,4,6,8,10,12,14,16 | column -s $'\t' -t > predictions.txt
#paste NN*/forward_pass.txt | cut -f 1,2,4,6,8,14,16 | column -s $'\t' -t > predictions.txt

#python3 /u/shubsharma/HDNNP/Naphthalene-Multi/predictions/testing-newstyle/script/post-prediction.py
