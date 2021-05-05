#!/bin/bash


# simple SLURM script for submitting G optimaztion job.


#-------------------------------------------------------


#set up parameters

#SBATCH -J launcher
#SBATHC -N 52
#SBATCH -n 832
#SBATCH -p normal
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err
#SBATCH -t 2-00:00:00



module load launcher

export LD_LIBRARY_PATH=/lib64:$LD_LIBRARY_PATH
export LAUNCHER_WORKDIR=$HOME/go
export LAUNCHER_JOB_FILE=jobfile

$LAUNCHER_DIR/paramrun
