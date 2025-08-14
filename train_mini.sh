set -euo pipefail


python main.py \
  --model_type "dna" \
  --router_type "default" \
  --dataset_name "roneneldan/TinyStories" \
  --dataset_config "default" \
  --batch_size 8 \
  --seq_len 64 \
  \
  --vocab_size 50257 \
  --d_model 192 \
  --n_heads 6 \
  --n_hops 4 \
  --n_modules 8 \
  --topk 2 \
  --capacity 32 \
  --mlp_mult 4 \
  --dropout 0.1 \
  --rope_base 10000.0 \
  \
  --router_temp 1.0 \
  --select_temp 1.0 \
  --gumbel_tau 1.0 \
  \
  --steps 500 \
  --warmup 100 \
  --lr_peak 8e-4 \
  --wd 0.1 \
  --clip 1.0 \
  --seed 42 \
  \
  --eval_every 100 \
  --log_every 10 \
  --eval_samples 4096 \
  --n_examples 2 \
  --gen_len 100 \
  --wandb_project "dna-cpu" \
