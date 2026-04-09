# MANAS v0.4 Non-Multitask Eval Batch

Last updated: 2026-04-06 18:40:44 IST

Launcher: `torchrun --standalone --nnodes=1 --nproc-per-node=8`

| Run | Status | W&B | Base Config | Run Dir | Trainer Log |
| --- | --- | --- | --- | --- | --- |
| v0.4 linear probe + attention pool | completed | [aws9y5ay](https://wandb.ai/neurodx-labs-neurodx/manas/runs/aws9y5ay) | `assets/conf/baseline/manas/manas_bench_v0p4_parentmemory.yaml` | `/share/arvasunewtemp/eval/manas_v0p4_parentmemory_lp_attn_20e_nomultitask` | ``/share/arvasunewtemp/eval/manas_v0p4_parentmemory_lp_attn_20e_nomultitask/log/baseline/manas-long-v0.4/torchrun_260406145051/manas-long-v0.4_trainer.log`` |
| v0.4 full fine-tune + attention pool | completed | [bv4efdtk](https://wandb.ai/neurodx-labs-neurodx/manas/runs/bv4efdtk) | `assets/conf/baseline/manas/manas_bench_v0p4_parentmemory.yaml` | `/share/arvasunewtemp/eval/manas_v0p4_parentmemory_fullft_attn_20e_nomultitask` | ``/share/arvasunewtemp/eval/manas_v0p4_parentmemory_fullft_attn_20e_nomultitask/log/baseline/manas-long-v0.4/torchrun_260406153852/manas-long-v0.4_trainer.log`` |
| v0.4 full fine-tune + avg pool | completed | [t5fix3j1](https://wandb.ai/neurodx-labs-neurodx/manas/runs/t5fix3j1) | `assets/conf/baseline/manas/manas_bench_v0p4_parentmemory.yaml` | `/share/arvasunewtemp/eval/manas_v0p4_parentmemory_fullft_avg_20e_nomultitask` | ``/share/arvasunewtemp/eval/manas_v0p4_parentmemory_fullft_avg_20e_nomultitask/log/baseline/manas-long-v0.4/torchrun_260406170952/manas-long-v0.4_trainer.log`` |
