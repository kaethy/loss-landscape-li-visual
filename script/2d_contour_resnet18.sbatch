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


$PYTHON plot_surface.py --x=-1:1:51 --y=-1:1:51 --model resnet18_torch \
--model_file cifar10/trained_nets/seed=2-epoch=191-val_loss=0.1278-model=ResNet18-optimizer=SGD-rho=0.05-adaptive=False-model_name=ResNet18_CIFAR10_seed2_SGD.pth \
--cuda --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --vmin 0.01 --vlevel 0.1 --plot