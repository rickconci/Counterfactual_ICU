#!/bin/bash
#SBATCH -J SDE_1   #! job name 
#SBATCH --output=logs/out_%A.out
#SBATCH --error=logs/err_%A.err
#SBATCH -A COMPUTERLAB-SL2-CPU
#SBATCH --time=6:00:00
#SBATCH -p icelake  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#! Optionally modify the environment seen by the application
#  (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc)
./etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded

echo "==================== Activating conda environment ===================="
conda activate expLight
echo "> Python version : $(which python)"
echo "==================== Run program ===================="
python Experiment_3/main_3.py --HPC_work

