#!/bin/tcsh
#SBATCH -t 3200
#SBATCH --nodes=1

#SBATCH --mail-user=blakehartley@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=End
date

cd ~/scratch/MC256_c_200/dat/
#module unload python
module load python
#module load python/2.7.8
#rm -rf ./dat/*.bin
#make clean
#make
#mpirun -n 4 ./3Dtransfer 1 700 1
#rm -rf ./vis/*
python plot6.py 50 407
#python mov_slice.py
hostname
date
