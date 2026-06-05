#!/bin/bash
#SBATCH --partition=gpu-a100-small
#SBATCH --gpus-per-task=1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem-per-gpu=8000MB
#SBATCH --time=4:00:00
#SBATCH --output=out.txt


cd /scratch/$USER/physicsnemo

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
apptainer exec --nv physicsnemo_26.03.sif \
python PINN/burgers_PPINO.py

apptainer exec --nv physicsnemo_26.03.sif \
python Burger_evaluate.py
# apptainer exec --nv physicsnemo_26.03.sif \
# python train.py


# apptainer exec --nv physicsnemo_26.03.sif \
# python evaluate.py
