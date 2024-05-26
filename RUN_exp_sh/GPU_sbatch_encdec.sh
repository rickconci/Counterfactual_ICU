#!/bin/bash
#! — SLURM HEADERS go at the top of the ﬁle 
#! Which project should be charged:
#SBATCH -A COMPUTERLAB-SL2-GPU 
#! How many whole nodes should be allocated?
#SBATCH --nodes=1 
#! How much wallclock time will be required?
#SBATCH --time=06:00:00 
#! Specify GPU nodes and number of GPUs. (Here: 4 GPUs on the ampere partition) 
#SBATCH -p ampere 
#SBATCH --gres=gpu:4
#! – Your custom commands to run the program echo "==================== Activating conda environment ===================="

conda init
conda activate expLight2

echo "==================== Run program ===================="

./run_experiments_enc_dec.sh
