#!/bin/bash

# this script is to move many similar named files to a saperate folder
# useful: as N2P2 generate bunch of files for each epochs its better to seperate them in a separate folders

echo "Moving files to folders"

mkdir weights trainforces sf-histogram testforces trainpoints testpoints nnp-train-log neuron-stats nnp-scaling-log
mv weights* weights/
mv trainforces* trainforces/
mv testforces* testforces/
mv trainpoints* trainpoints/
mv testpoints* testpoints/
mv nnp-train.log* nnp-train-log/
mv neuron-stats* neuron-stats/
mv sf.* sf-histogram/
mv nnp-scaling.log* nnp-scaling-log/
