#!/bin/tcsh

#SBATCH
#SBATCH -t 24:0:0
#SBATCH --partition=gpuk80
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=15900
#SBATCH --exclusive

#SBATCH --gres=gpu:1

#SBATCH --mail-user=blakehartley@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

date

module load cuda openmpi

cd /home-1/bth\@umd.edu/scratch/MC256_CP/bin/

#rm -rf ./dat/hydrogen/*.bin ./dat/helium/*.bin
#rm -rf ./dat/photons/*.bin ./dat/background/*.bin
#make clean
#make
mpirun -n 8 ./arc

hostname
date
