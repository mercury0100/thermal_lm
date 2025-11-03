⚡ main ~/thermal_lm python train_gpt2.py --variant thermal
🔥 Running THERMAL variant on gpt2 | device=cuda
[Checkpoint] Saved to checkpoints/thermal_best_ep1
[Epoch 1] loss=3.5349 | ppl=30.86 | entropy=3.325 | tau_mean=1.08588 | corr(ent,tau)=0.315
[Checkpoint] Saved to checkpoints/thermal_best_ep2
[Epoch 2] loss=3.1572 | ppl=30.40 | entropy=3.170 | tau_mean=1.05190 | corr(ent,tau)=0.344

⚡ main ~/thermal_lm python train_gpt2.py --variant baseline
🔥 Running BASELINE variant on gpt2 | device=cuda
[Checkpoint] Saved to checkpoints/baseline_best_ep1
[Epoch 1] loss=3.5274 | ppl=30.78 | entropy=3.330
[Checkpoint] Saved to checkpoints/baseline_best_ep2
[Epoch 2] loss=3.1442 | ppl=30.49 | entropy=3.158

⚡ main ~/thermal_lm python train_llama.py \
  --base_model meta-llama/Llama-3.2-1B \
  --train_dataset c4-small \
  --use_lora --lora_rank 8 --lora_lr 1e-4 \
  --variant thermal --epochs 2 \
  --batch_size 1 --grad_accum_steps 32 \
  --max_length 256 --lambda_reg 1e-3
🔥 Running THERMAL | base=meta-llama/Llama-3.2-1B | data=c4-small | device=cuda
[Data] Loading c4-small with streaming: train≈10000, val≈1000
[Data] Train batches: 10000 | Val batches: 1000
[Checkpoint] Saved to checkpoints/thermal_best_ep1
[Epoch 1] loss=3.0229 | ppl=21.14 | entropy=3.111 | tau_mean=0.97964 | corr(ent,tau)=0.271
[Checkpoint] Saved to checkpoints/thermal_best_ep2
[Epoch 2] loss=3.0057 | ppl=21.10 | entropy=3.062 | tau_mean=0.95923 | corr(ent,tau)=0.379

⚡ main ~/thermal_lm python train_llama.py \
  --base_model meta-llama/Llama-3.2-1B \
  --train_dataset c4-small \
  --use_lora --lora_rank 8 --lora_lr 1e-4 \
  --variant baseline --epochs 5 \
  --batch_size 1 --grad_accum_steps 32 \
  --max_length 256 --lambda_reg 1e-3
🔥 Running BASELINE | base=meta-llama/Llama-3.2-1B | data=c4-small | device=cuda
[Data] Loading c4-small with streaming: train≈10000, val≈1000
[Data] Train batches: 10000 | Val batches: 1000
[Checkpoint] Saved to checkpoints/baseline_best_ep1
[Epoch 1] loss=3.0102 | ppl=20.98 | entropy=3.080
[Epoch 2] loss=2.9999 | ppl=20.99 | entropy=3.077

python train_gpt2.py --variant thermal --epochs 2 --batch_size 2 --alpha_entropy 5e-2 --lambda_reg 1e-2 

