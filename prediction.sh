#!/bin/bash
#SBATCH  --output=/srv/beegfs02/scratch/aegis_guardian/data/Timothy/logs/deep_sanity_predict%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --job-name=deepSanity



source /itet-stor/zahmad/net_scratch/conda/etc/profile.d/conda.sh
conda activate tf-env

python prediction.py
"$@"