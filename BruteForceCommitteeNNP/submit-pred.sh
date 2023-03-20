#!/bin/bash -l
## run in /bin/bash NOTE the -l flag!
#
## do not join stdout and stderr
#SBATCH -o job.out
#SBATCH -e job.err
## name of the job
#SBATCH -J prediction-NN1 
## execute job from the current working directory
## this is default slurm behavior
#SBATCH -D ./
## do not send mail
#SBATCH --mail-type=NONE
#SBATCH --ntasks=1         # launch job on a single core
#SBATCH --cpus-per-task=1  #   on a shared node
#SBATCH --mem=2000MB       # memory limit for the job
#SBATCH --time=0:20:00

module purge
module load intel/21.6.0 impi/2021.6  mkl/2022.1 fftw-mpi/3.3.10 
module load anaconda/3/5.1   

conda activate n2p2

export OMP_NUM_THREADS=1

python3 LinearRegression_concat.py > log.prediction 

