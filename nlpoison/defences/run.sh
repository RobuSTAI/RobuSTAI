#!/bin/bash -l
#SBATCH --output=/users/k20095632/projects/RobuSTAI/nlpoison/defences/ChenHSBert.out     #change the "bert_" part every new run
#SBATCH --time=03:00:00
#SBATCH --mem=32768
#SBATCH --job-name=chenstuff
#SBATCH --ntasks=1
echo "Hello, World From $HOSTNAME"
module load libs/cuda
srun  python3 defense_AC_run.py
echo "Goodbye, World! From $HOSTNAME"
