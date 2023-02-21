#!/bin/bash

#SBATCH --account phwq4930-renal-canc
#SBATCH --qos epsrc
#SBATCH --time 0-12:00:00
#SBATCH --tasks-per-node 1
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 80G


set -x

module purge; module load baskerville
module load bask-apps/test
module load Miniconda3/4.10.3
conda init bash
source /bask/homes/r/ropj6012/.bashrc

# Location of conda environment
CONDA_ENV_PATH="/bask/projects/p/phwq4930-renal-canc/conda_env/ovseg_env"

# Activate the environment
conda activate "${CONDA_ENV_PATH}"

# Run jobs

python gpu_manager_prep.py 1 stage1_prep_Xmm_HPC_allbinary.py coreg_ncct 1