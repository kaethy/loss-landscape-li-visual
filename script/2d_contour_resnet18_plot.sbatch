#!/usr/bin/env bash
#SBATCH --gres gpu:1
#SBATCH --mem 8GB
#SBATCH --cpus-per-task 4
#SBATCH --time 48:00:00
#SBATCH --constrain "galadriel"
#SBATCH --output /Midgard/home/%u/thesis/logs/%J_slurm.out
#SBATCH --error  /Midgard/home/%u/thesis/logs/%J_slurm.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"

PYTHON=/Midgard/home/kadec/miniforge3/envs/livi/bin/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Midgard/home/kadec/miniforge3
export OMPI_MCA_opal_cuda_support=true


$PYTHON plot_2D.py --surf_file cifar10/trained_nets/seed=3-epoch=193-val_loss=0.1898-model=ResNet18-optimizer=SGD-rho=0.05-adaptive=False-model_name=ResNet18_CIFAR10_seed3_SGD.pth_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-1.0,1.0,51]x[-1.0,1.0,51].h5 --surf_name train_loss --vmin 0.001 --vlevel 0.02 --vmax 4