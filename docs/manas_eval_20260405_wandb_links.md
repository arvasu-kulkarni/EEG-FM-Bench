# MANAS Eval Batch

Last updated: 2026-04-06 03:03:17 IST

Launcher: `torchrun --standalone --nnodes=1 --nproc-per-node=8`

| Run | Status | W&B | Base Config | Run Dir | Trainer Log |
| --- | --- | --- | --- | --- | --- |
| v0.4 linear probe + attention pool | completed | [egjabl48](https://wandb.ai/neurodx-labs-neurodx/manas/runs/egjabl48) | `assets/conf/baseline/manas/manas_bench_v0p4_parentmemory.yaml` | `/share/arvasunewtemp/eval/manas_v0p4_parentmemory_mt_lp_attn_20e` | `/share/arvasunewtemp/eval/manas_v0p4_parentmemory_mt_lp_attn_20e/log/baseline/manas-long-v0.4/torchrun_260405205623/manas-long-v0.4_trainer.log` |
| v0.4 linear probe + avg pool | completed | [47uu7uc0](https://wandb.ai/neurodx-labs-neurodx/manas/runs/47uu7uc0) | `assets/conf/baseline/manas/manas_bench_v0p4_parentmemory.yaml` | `/share/arvasunewtemp/eval/manas_v0p4_parentmemory_mt_lp_avg_20e` | `/share/arvasunewtemp/eval/manas_v0p4_parentmemory_mt_lp_avg_20e/log/baseline/manas-long-v0.4/torchrun_260405215624/manas-long-v0.4_trainer.log` |
| v0.4 full fine-tune + attention pool | completed | [krjiryoa](https://wandb.ai/neurodx-labs-neurodx/manas/runs/krjiryoa) | `assets/conf/baseline/manas/manas_bench_v0p4_parentmemory.yaml` | `/share/arvasunewtemp/eval/manas_v0p4_parentmemory_mt_fullft_attn_20e` | `/share/arvasunewtemp/eval/manas_v0p4_parentmemory_mt_fullft_attn_20e/log/baseline/manas-long-v0.4/torchrun_260405225624/manas-long-v0.4_trainer.log` |
| v0.2 linear probe + attention pool | completed | [xkonu2yk](https://wandb.ai/neurodx-labs-neurodx/manas/runs/xkonu2yk) | `assets/conf/baseline/manas/manas_bench_v0p2_scaleattn_localaux_multitask_eval.yaml` | `/share/arvasunewtemp/eval/manas_v0p2_scaleattn_localaux_mt_lp_attn_20e` | `/share/arvasunewtemp/eval/manas_v0p2_scaleattn_localaux_mt_lp_attn_20e/log/baseline/manas-long-v0.2/torchrun_260406010126/manas-long-v0.2_trainer.log` |
| v0.2 linear probe + avg pool | completed | [22s1udjm](https://wandb.ai/neurodx-labs-neurodx/manas/runs/22s1udjm) | `assets/conf/baseline/manas/manas_bench_v0p2_scaleattn_localaux_multitask_eval.yaml` | `/share/arvasunewtemp/eval/manas_v0p2_scaleattn_localaux_mt_lp_avg_20e` | `/share/arvasunewtemp/eval/manas_v0p2_scaleattn_localaux_mt_lp_avg_20e/log/baseline/manas-long-v0.2/torchrun_260406020226/manas-long-v0.2_trainer.log` |
