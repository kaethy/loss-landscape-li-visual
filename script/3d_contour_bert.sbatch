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


$PYTHON plot_3D.py --surf_files script/plot_files/bert_files_adam_sam_asam.txt --surf_name train_loss --vmin 0.15 --vlevel 0.03 --vmax 0.8 --labels AdamW SAM ASAM --save_dir /Midgard/home/kadec/loss-landscape-li-visual/script/plot_files/