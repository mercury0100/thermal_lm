⚡ ~/thermal_lm python train_thermal_lm.py --variant thermal           
🔥 Running THERMAL variant on distilgpt2 | device=cuda
[Checkpoint] Saved to checkpoints/thermal_best_ep1
[Epoch 1] loss=3.7901 | ppl=38.85 | entropy=3.481 | tau_mean=1.08410 | corr(ent,tau)=0.250
[Checkpoint] Saved to checkpoints/thermal_best_ep2
[Epoch 2] loss=3.4529 | ppl=38.00 | entropy=3.341 | tau_mean=1.05678 | corr(ent,tau)=0.245

⚡ ~/thermal_lm python train_thermal_lm.py --variant baseline
🔥 Running BASELINE variant on distilgpt2 | device=cuda
[Checkpoint] Saved to checkpoints/baseline_best_ep1
[Epoch 1] loss=3.7833 | ppl=38.81 | entropy=3.477
[Checkpoint] Saved to checkpoints/baseline_best_ep2
[Epoch 2] loss=3.4436 | ppl=38.09 | entropy=3.324