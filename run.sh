#!/bin/bash
# shellcheck disable=SC2034
# -name
time=$(date "+%Y_%m_%d-%H_%M_%S")
project_name="miniGPT-${time}"
echo "${time}"

python main.py \
--mode "train" \
--use_model "miniGPT" \
--n_head 12 \
--load_model  "" \
--proj_dir "test" \
--data_file "" \
--data_type "minipile" \
--att_type "New_RoPE" \
--vocab_size 0 \
--ctx_len 512 \
--epoch_steps 50000 \
--epoch_count 4 \
--epoch_begin 0 \
--epoch_begin_steps 0 \
--epoch_save 2000 \
--micro_bsz 16 \
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
--devices 1 \
--precision bf16 \
--strategy ddp_find_unused_parameters_False \
#--strategy ddp_find_unused_parameters_False \


