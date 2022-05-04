#!/bin/bash
#SBATCH  --output=/srv/beegfs02/scratch/aegis_guardian/data/Timothy/logs/deep_train%j.out
#SBATCH  --gres=gpu:6
#SBATCH  --job-name=deepTrain



source /itet-stor/zahmad/net_scratch/conda/etc/profile.d/conda.sh
conda activate tf-env

python train.py
"$@"