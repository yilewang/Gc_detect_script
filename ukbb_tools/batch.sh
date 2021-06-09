#!/bin/bash

case=$(ls ~/HSAM/hsam/)


for i in $case
do
	cd ~/HSAM/hsam/$i
	3dAFNItoNIFTI struct+orig.
	gzip struct.nii
	${FSLDIR}/bin/fslreorient2std struct.nii.gz
	mv struct.nii.gz "$i"_T1.nii.gz
done
