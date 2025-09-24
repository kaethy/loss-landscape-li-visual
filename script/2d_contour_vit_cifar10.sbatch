#!/usr/bin/env bash
#SBATCH --gres gpu:1
#SBATCH --mem 8GB
#SBATCH --cpus-per-task 4
#SBATCH --time 48:00:00
#SBATCH --constrain "arwen"
#SBATCH --output /Midgard/home/%u/thesis/logs/%J_slurm.out
#SBATCH --error  /Midgard/home/%u/thesis/logs/%J_slurm.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"

PYTHON=/Midgard/home/kadec/miniforge3/envs/livi/bin/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Midgard/home/kadec/miniforge3
export OMPI_MCA_opal_cuda_support=true


$PYTHON plot_surface.py --x=-1:1:51 --y=-1:1:51 --model vit_10 \
--model_file /local_storage/users/kadec/experiment_results/CIFAR10_ViT__SAM/seed=0-epoch=06-val_loss=0.0316-model=ViT-optimizer=AdamW-rho=0.07341469528971963-adaptive=False-model_name=ViT_CIFAR10_seed0_AdamW.pth \
--dataset cifar10_vit --cuda --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --vmin 0.01 --vlevel 0.1 --plot
