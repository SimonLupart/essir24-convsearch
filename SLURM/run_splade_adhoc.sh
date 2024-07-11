#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=adhoc
#SBATCH --partition gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-01:00:00
#SBATCH --mem=120gb #120gb
#SBATCH -c 4
#SBATCH --output=/projects/0/prjs0871/hackathon/conv-search/EXP/log_slurm/adhoc-%j.out

# Set-up the environment.
conda activate cs
nvidia-smi

# depending on the type of queries you use, either only the last query or all the previous conversation
mode=last # last all
base_dir=/projects/0/prjs0871/hackathon/conv-search/

index_dir=$base_dir/DATA/topiocqa_index
eval_queries=$base_dir/DATA/queries_rowid_dev_$mode.tsv

python -m splade.retrieve init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil \
           config.pretrained_no_yamlconfig=true config.index_dir="$index_dir" \
           config.out_dir="$base_dir/EXP/out_adhoc/$mode/" \
           data.Q_COLLECTION_PATH=[$eval_queries] \
           data.EVAL_QREL_PATH=[$base_dir/DATA/qrel_rowid_dev.json]

# python -m splade.evaluate init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil \
#            config.pretrained_no_yamlconfig=true config.index_dir="$index_dir" \
#            config.out_dir="$base_dir/EXP/out_adhoc/$mode/" \
#            data.Q_COLLECTION_PATH=[$eval_queries] \
#            data.EVAL_QREL_PATH=[$base_dir/DATA/qrel_rowid_dev.json]
