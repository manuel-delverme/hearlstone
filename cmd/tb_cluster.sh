#!/usr/bin/env bash
#SBATCH -J hs
#SBATCH -p high
#SBATCH -N 1
#SBATCH -n 1

#SBATCH --workdir=/homedtic/stotaro/hearlstone
#SBATCH -o /homedtic/stotaro/hearlstone/tb_logs/%N.%J.out # STDOUT
#SBATCH -e /homedtic/stotaro/hearlstone/tb_logs/%N.%j.err # STDERR
#SBATCH -C intel #request intel node (those have infiniband)

set -x

module load PyTorch/1.0.0-foss-2017a-Python-3.6.4-CUDA-9.0.176
source /homedtic/stotaro/hs/bin/activate
tensorboard --logdir hearlstone/ppo_log/tensorbaord --port 6006
