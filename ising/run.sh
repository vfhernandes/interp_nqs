#!/bin/sh
#
#SBATCH --job-name="nqs"
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --account=research-as-qn


module load 2023r1
module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate nqs

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python run.py $1

conda deactivate
