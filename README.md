⚡ main ~/thermal_lm python train_gpt2.py --variant thermal --lambda_reg 0.
🔥 Running THERMAL variant on cuda
[Epoch 1] FreeEnergy=3.5518 | PPL=31.26 | Entropy=2.809 | TauMean=1.0950 | Corr(ent,tau)=0.198
[Epoch 2] FreeEnergy=3.1686 | PPL=30.92 | Entropy=2.857 | TauMean=1.0531 | Corr(ent,tau)=0.211
[Epoch 3] FreeEnergy=2.9705 | PPL=31.37 | Entropy=2.920 | TauMean=1.0118 | Corr(ent,tau)=0.262
⚡ main ~/thermal_lm python train_gpt2.py --variant baseline
🔥 Running BASELINE variant on cuda
[Epoch 1] FreeEnergy=3.5423 | PPL=31.15 | Entropy=3.333 | TauMean=0.0000 | Corr(ent,tau)=0.000
[Epoch 2] FreeEnergy=3.1534 | PPL=31.04 | Entropy=3.150 | TauMean=0.0000 | Corr(ent,tau)=0.000
[Epoch 3] FreeEnergy=2.9527 | PPL=31.46 | Entropy=3.008 | TauMean=0.0000 | Corr(ent,tau)=0.000