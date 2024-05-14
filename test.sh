#!/bin/bash
# shellcheck disable=SC2034
# -name
export CXX=g++
time=$(date "+%Y_%m_%d-%H_%M_%S")
project_name="miniGPT-${time}"
echo "${time}"
set -x
export TORCH_DISTRIBUTED_DEBUG=INFO
export HF_ENDPOINT=https://hf-mirror.com

python main.py \
--mode "test" \
--use_model "miniGPT" \
--n_head 12 \
--load_model  "test/rope/New_RoPEepoch=3-step=124000.ckpt" \
--proj_dir "test" \
--data_file "/home/qingyu_yin/data/RWKV/pile/pile/train" \
--data_type "minipile" \
--att_type "New_RoPE" \
--vocab_size 0 \
--ctx_len 512 \
--epoch_steps 50000 \
--epoch_count 4 \
--epoch_begin 0 \
--epoch_begin_steps 0 \
--epoch_save 2000 \
--micro_bsz 1 \
--n_layer 12 \
--n_embd 768 \
--lr_init 5e-4 \
--lr_final 5e-5 \
--warmup_steps 4000 \
--beta1 0.9 \
--beta2 0.98 \
--adam_eps 1e-8 \
--accelerator gpu \
--num_nodes 1 \
--devices 2 \
--precision fp16 \
--strategy ddp_find_unused_parameters_False \
#--strategy ddp_find_unused_parameters_False \


