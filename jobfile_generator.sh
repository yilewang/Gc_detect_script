#!/bin/bash
makedir=$(ls /mnt/c/Users/Wayne/workdir/zip/AD_conn/ | cut -c1-5 )

for i in $makedir
do 
    seq -f "python go_sim_hpc.py AD $i %g" 0.01 0.001 0.06 >> jobfile
done
