#!/usr/bin/env bash
#SBATCH -J hs
#SBATCH -p high
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem-per-cpu 6000

#SBATCH --workdir=/homedtic/stotaro/hearlstone
#SBATCH -o /homedtic/stotaro/hearlstone/jobs/%N.%J.hs.out # STDOUT
#SBATCH -e /homedtic/stotaro/hearlstone/jobs/%N.%J.hs.err # STDOUT

set -x
module load PyTorch/1.1.0-foss-2017a-Python-3.6.4-CUDA-9.0.176
source ~/hs/bin/activate
for i in {0..5}
do
  telegram-send "Launching training in the cluster for $1. Connecting at $2"
  python -O main.py --comment $1 --address $2
  telegram-send "Launch failed. Trial $i/5. Retrying in 30s"
  sleep 30s
done
