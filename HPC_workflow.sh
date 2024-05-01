
# !/bin/bash
# SBATCH -J job_name
# SBATCH --output=logs/out_%A.out
# SBATCH --error=logs/err_%A.err
# SBATCH -A COMPUTERLAB-SL2-CPU
# SBATCH --time=2:00:00
# SBATCH -p icelake
# SBATCH --nodes=1
# SBATCH --ntasks-per-node=1

# ! Optionally modify the environment seen by the application
# ! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc)
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load miniconda/3              # REQUIRED - loads the basic environment

python main_3.py 

sinfo


sintr -t 1:0:0 --gres=gpu:1 -A YOURPROJECT-GPU -p ampere


sintr -t 00:20:00 --nodes=1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=3 --partition=ampere -A COMPUTERLAB-SL3-GPU --qos=INTR

sintr -t 00:20:00 --nodes=1 --gres=gpu:1 --ntasks-per-node=1 --partition=ampere -A COMPUTERLAB-SL3-GPU --qos=INTR