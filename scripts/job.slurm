#!/bin/bash

#SBATCH --time=00:05:00
#SBATCH -p gpu -C gpu2v100 --gres=gpu:1
#SBATCH -o test.o%j

module load cuda
module swap intel gcc
module load python/3.7.0


cd $PFSDIR