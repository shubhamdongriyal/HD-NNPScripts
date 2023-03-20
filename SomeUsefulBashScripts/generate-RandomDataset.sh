#!/bin/bash

# this script generate N random splitted set of a dataset 
# we use it to generate ensemble of neural networks

cd trainset-80%-16
cp ../input.data .
cp input.data input.data-orig

RANDOM=$$
for i in `seq 16`
do
	echo Random seed:  $RANDOM
	nnp-select random 0.8 $RANDOM
	mkdir data-$i
	mv output.data data-$i
	mv reject.data data-$i
	mv nnp-select.log data-$i

	cp ../input.nn data-$i
	wait

	cd data-$i
	mv output.data input.data
	
	echo computing symmetry function for the dataset with seed: $RANDOM
	mpirun -np 4 nnp-scaling 500

	cd ../
done

mv input.data-orig input.data
