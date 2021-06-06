#!/bin/bash

case=$(ls ~/HSAM/hsam/)

mkdir $case

for i in $case
do
	cd ~/HSAM/$i
	mv ~/HSAM/hsam/$i/"$i"_T1.nii.gz .
done
