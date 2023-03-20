#!/bin/bash

cp input.data-144 input.data-orig
echo 'splitting the file'
split -l 151 -a 6 input.data-144


SPLITFILES='x*'
count=1

for i in $SPLITFILES 
do
	mv $i input.data
	mkdir frame.$count
	mv input.data frame.$count
	((count++))
done
