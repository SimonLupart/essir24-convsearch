#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=ft
#SBATCH --partition gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-10:00:00
#SBATCH --mem=120gb #120gb
#SBATCH -c 4
#SBATCH --output=/projects/0/prjs0871/hackathon/conv-search/EXP/log_slurm/ft-%j.out

# Set-up the environment.
conda activate splade
nvidia-smi

# Train the model
port=$(shuf -i 29500-29599 -n 1)
config=config_hf_splade_TOPIOCQA.yaml

base_dir=/projects/0/prjs0871/hackathon/conv-search/

torchrun --nproc_per_node 1 --master_port $port -m splade.hf_train --config-name=$config \
         config.checkpoint_dir="$base_dir/EXP/out_ft/ckpt_splade_ft/" \
         data.TRAIN.DATASET_PATH=$base_dir/DATA/run.json \
         data.TRAIN.D_COLLECTION_PATH=$base_dir/DATA/full_wiki_segments.tsv \
         data.TRAIN.Q_COLLECTION_PATH=$base_dir/DATA/queries_all.tsv \
         data.TRAIN.QREL_PATH=$base_dir/DATA/qrel.json

# Inference
index_dir=$base_dir/DATA/topiocqa_index
eval_queries=$base_dir/DATA/queries_rowid_dev_all.tsv
python -m splade.retrieve --config-name=$config config.checkpoint_dir="$base_dir/EXP/out_ft/ckpt_splade_ft/" \
        config.index_dir="$index_dir" config.out_dir="$base_dir/EXP/out_ft/" \
        data.Q_COLLECTION_PATH=[$eval_queries] \
        data.EVAL_QREL_PATH=[$base_dir/DATA/qrel_rowid_dev.json]


# # python  -m splade.index --config-name=config_hf_splade_16neg_nodistil_TOPIOCQA_b_2e_gen_train_v3.yaml config.checkpoint_dir="/projects/0/prjs0871/splade/EXP/config_hf_splade_16neg_nodistil_TOPIOCQA_b_2e_gen_train_v3/" config.index_dir="/projects/0/prjs0871/splade/EXP/config_hf_splade_16neg_nodistil_TOPIOCQA_b_2e_gen_train_v3/index"
# python -m splade.flops --config-name=config_hf_splade_16neg_nodistil_TOPIOCQA_b_2e_frozen_256.yaml config.checkpoint_dir="/projects/0/prjs0871/essir-conv//EXP/config_hf_splade_16neg_nodistil_TOPIOCQA_b_2e_frozen_256/" config.index_dir="/projects/0/prjs0871/splade/EXP/config_hf_splade_32neg_nodistil_TOPIOCQA_b_2e_frozen/index" config.out_dir="/projects/0/prjs0871/essir-conv//EXP/config_hf_splade_16neg_nodistil_TOPIOCQA_b_2e_frozen_256/out/"
