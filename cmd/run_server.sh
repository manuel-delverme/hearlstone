#!/usr/bin/env bash
#SBATCH -J sb
#SBATCH -p high
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --mem-per-cpu 6000

#SBATCH --workdir=/homedtic/stotaro/hearlstone
#SBATCH -o /homedtic/stotaro/hearlstone/%N.%J.sb.out # STDOUT
#SBATCH -e /homedtic/stotaro/hearlstone/%N.%J.sb.err # STDOUT

set -x
module load dotNET-SDK/2.2.6-foss-2017a
dotnet run -p pysabberstone/dotnet_core rpc $1
