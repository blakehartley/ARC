#!/bin/tcsh
#SBATCH -A ricotti-prj-aac
#SBATCH -t 48:0:0
#SBATCH -n 8
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gpus=a100
#SBATCH --mem-per-cpu=8000

#SBATCH --mail-user=blakehartley@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

date

#module unload gcc
#module load gcc/8.4.0 openmpi/gcc/8.4.0 cuda/gcc/8.4.0
module load openmpi cuda

#cd /scratch/zt1/project/ricotti-prj/user/bth/MC64_C/src/
cd /home/bth/scratch/MC256_c_200/src/

#make clean
#make

#cd /scratch/zt1/project/ricotti-prj/user/bth/MC64_C/bin/
#cd /home/bth/scratch/MC256_C/bin/
#rm -rf ./dat/hydrogen/*.bin ./dat/helium/*.bin
#rm -rf ./dat/photons/*.bin ./dat/background/*.bin
mpirun -n 8 ../bin/arc
#mpirun -n 8 ./arc > ../output.dat
#mpirun -n 1 ./3Dtransfer > ./test/output1.dat
#mpirun -n 32 ./3Dtransfer > output32.dat
#mpirun -n 4 ./3Dtransfer > ./test/output4.dat
#mpirun -n 8 ./3Dtransfer > ./test/output8.dat
#mpirun -n 16 ./3Dtransfer > ./test/output16.dat
#mpirun -n 32 ./3Dtransfer > ./test/output32.dat

#cd /lustre/bth/MC256_T1/dat/
#module unload python
#module load python/2.7.8
#rm -rf ./dat/*.bin
#make clean
#make
#mpirun -n 4 ./3Dtransfer 1 700 1
#rm -rf ./vis/*
#python plot6.py 14
#python mov_slice.py

hostname
date
