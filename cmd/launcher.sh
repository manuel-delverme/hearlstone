#!/usr/bin/env bash
#SBATCH -J general
#SBATCH -p high
#SBATCH -N 1
#SBATCH -n 1

#SBATCH --workdir=/homedtic/stotaro/hearlstone
#SBATCH -o /homedtic/stotaro/hearlstone/%N.%J.general.out # STDOUT
#SBATCH -e /homedtic/stotaro/hearlstone/%N.%J.general.err # STDOUT

set -x
PID=$(sbatch run_server.sh $1 | grep -Eo '[[:digit:]]*')
sleep 1m
NODE=$(squeue --job $PID | grep -e "node\w*" -o)
sbatch ./train_cluster.sh $2 $NODE:$PID
