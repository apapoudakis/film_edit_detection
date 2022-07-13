#!/bin/bash

#SBATCH --time=00:05:00
#SBATCH -p gpu -C gpu2v100 --gres=gpu:1
#SBATCH -o /home/$USER/test.o%j

# from gallina home to tmp folder
rsync -az hpc3:/mnt/rds/redhen/gallina/home/axp1083/ /tmp/axp1083/

cd /tmp/$USER/
git clone https://github.com/apapoudakis/film_edit_detection.git

module load gcc/6.3.0
export PYTHONPATH="${PYTHONPATH}:/tmp/axp1083/film_edit_detection"
cd film_edit_detection
pip3 install -r requirements.txt

# from tmp folder to gallina home
rsync -az /tmp/axp1083/ hpc3:/mnt/rds/redhen/gallina/home/axp1083/Data/

rm -f -r /tmp/$USER/
