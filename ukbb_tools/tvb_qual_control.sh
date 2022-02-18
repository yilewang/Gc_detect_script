#!/bin/bash

#this file checks that all the appropriate files are there from the resulting processing of the T1 through the uk-bb pipeline
#to run, type bash tvb_qual_control CASE#1 CASE#2 ...
#where the arguments are the names of the folders containing the data
#the script should be located two levels above the T1 weighted

#to print the results to a file add "> file_name" at the end of the arguments

declare -a FILES_ARRAY

FILES_ARRAY=("T1_fast" "T1_first" "T1_sienax" "T1_vbm" "transforms" "cort_subcort_GM.nii.gz" "GMatlas_to_T1.nii.gz" "labelled_GM.nii.gz" "labelled_GM.niml.lt" "labelled_GMI.nii.gz" "labelled_GMI.niml.lt" "T1_brain_mask.nii.gz" "T1_brain_to_MNI.nii.gz" "T1_brain.nii.gz" "T1_orig_defaced.nii.gz" "T1_orig_QC_CNR_lower.txt" "T1_orig_QC_CNR_upper.txt" "T1_orig_ud.nii.gz" "T1_orig.nii.gz" "T1_QC_COG.txt" "T1_QC_face_mask_inside_brain_mask.txt" "T1_unbiased_brain.nii.gz" "T1_unbiased.nii.gz" "T1.nii.gz")

for CASE in "$@"
do
	echo "$CASE"
for NAME in "${FILES_ARRAY[@]}"
do
	SLASH="/"
	if [ ! -f "${CASE}${SLASH}T1${SLASH}${NAME}" ] && [ ! -d "${CASE}${SLASH}T1${SLASH}${NAME}" ]; then
		echo "${NAME} not found "	
	fi
done

done

