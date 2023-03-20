#!bin/bash

######### foreach script for making the input.data files used for multiple predictions #######

# n2p2 predictions for every test structure
# as n2p2 predicts only one structure at a time
# this code is splitting your test data into separate files for each configuration which is used for predictions (nnp-predict)
# it also generate a n2p2_energy_frames.out file where you can see all the predictions for your test data

######### . No need to use below code for now ##############

#mkdir nn-mulprediction
#cp foreach-nnMulprediction.sh 
#cd nn-mulprediction
#echo 'moved to a new folder so that YOU DONT MESS UP'
#echo 'splitting the input.data file into 100 files'
#echo 'Original input.data file is now in input.data-orig'

################ Main script strats from here ########################

cp test.data input.data-orig
split -l 200 -a 4 test.data  #splitting the large file to small files 

mkdir mulPredict
wait

SPLITFILES='x*'
count=1
for i in $SPLITFILES
do
	#echo  'reading this file' $i
	#echo $count
	mv $i input.data
	wait
	nnp-predict structure.out
	wait
	mv input.data input.data.$count
	mv values.001.out values.001.out.$count
	mv values.008.out values.008.out.$count
	sed '14!d' energy.out >> n2p2_energy_frames.out
	mv energy.out energy.out.$count
	mv nnatoms.out nnatoms.out.$count
	mv nnforces.out nnforces.out.$count
	mv output.data output.data.$count
	mv nnp-predict.log nnp-predict.log.$count

	mkdir nnp-predict.$count
	mv *.$count nnp-predict.$count

	((count++))
done
wait

mv nnp-predict* mulPredict
