#!/bin/bash

#SBATCH --account phwq4930-renal-canc
#SBATCH --qos epsrc
#SBATCH --time 0-23:59:00
#SBATCH --nodes 1
#SBATCH --gpus 4
#SBATCH --cpus-per-gpu 2
#SBATCH --mem 120G

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

python gpu_manager_stage2.py 4 stage2_train_Xmm_HPC.py coreg_ncct 6 1 "0,1,2,3"
