#!/usr/bin/bash


cd $1
case=$(ls $1)

for i in $case
do
        tmp=$(cut -d"." -f1 <<< $i)
        tar -xf $i && mv raw $tmp
        cd $tmp
        mkdir "$tmp"_output
        # for T1 weight
        t1_par=$(find $1/$tmp/ -iname "*HiRes*.PAR.gz")
        t1_rec=$(find $1/$tmp/ -iname "*HiRes*.REC.gz")
        # for resting state fMRI
        resting_par=$(find $1/$tmp/ -iname "*6_1.PAR*")
        resting_rec=$(find $1/$tmp/ -iname "*6_1.REC*")
        # for DTI
        dti_par=$(find $1/$tmp/ -iname "*DTI*.PAR.gz")
        dti_rec=$(find $1/$tmp/ -iname "*DTI*.REC.gz")
        # unzip each files
        gzip -d $t1_par $t1_rec $resting_par $resting_rec $dti_par $dti_rec
        # create names for dcm2niix
        t1=$(find $1/$tmp/ -iname "*HiRes*.PAR" -exec basename {} ';')
        resting=$(find $1/$tmp/ -iname "*6_1.PAR" -exec basename {} ';')
        dti=$(find $1/$tmp/ -iname "*DTI*.PAR" -exec basename {} ';')
        #base2=$(find $1/$tmp/ -iname "*4_1.PAR" -exec basename \{} .PAR.gz \;)
        dcm2niix -i y -z y -f "$tmp"_T1 -o $1/$tmp/"$tmp"_output $t1
        dcm2niix -i y -z y -f "$tmp"_resting_fmri -o $1/$tmp/"$tmp"_output $resting
        dcm2niix -i y -z y -f "$tmp"_dti -o $1/$tmp/"$tmp"_output $dti
        cd ..
done

cd
hsam_out=$(find ~/hsam_raw -iname "*_output")
mv $hsam_out ~/hsam_data
