#!/bin/bash

#SBATCH --account phwq4930-renal-canc
#SBATCH --qos epsrc
#SBATCH --time 0-06:00:00
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 1

set -x

module purge; module load baskerville
module load bask-apps/test
module load Miniconda3/4.10.3

# Location for storage the conda environment
CONDA_ENV_PATH="/bask/projects/p/phwq4930-renal-canc/conda_env/ovseg_env"

# Create the environment. Only required once.
conda config --add pkgs_dirs "${CONDA_ENV_PATH}"/pkgs
conda create --yes --prefix "${CONDA_ENV_PATH}"
conda init bash
source ~/.bashrc

# Activate the environment
conda activate "${CONDA_ENV_PATH}"
# Choose your version of Python
conda install --yes python=3.10.9 pip=22.3.1

# Continue to install any further items as required.
# For example:
conda install --yes numpy=1.23.5 matplotlib=3.6.2 scikit-image=0.19.3
conda install --yes -c conda-forge nibabel=5.0.0 pydicom=2.3.1 tqdm=4.64.1
conda install --yes pytorch=1.13.1 torchvision=0.14.1 torchaudio=0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install rt_utils==1.2.7

