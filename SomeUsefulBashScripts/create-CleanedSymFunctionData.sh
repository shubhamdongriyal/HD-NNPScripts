#!/bin/bash

# this script creates a cleaned symmetry function data file used for postprocessing

#cd dataset
for i in data-{1..16} 
do
	cd $i
	awk '{if (NF > 26) print $0}' function.data > function.cleaned.data
	grep 'energy' input.data | awk '{print $2}' > energy.data
	cd ../
done
