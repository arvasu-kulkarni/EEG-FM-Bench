# MANAS BCIC/Workload/Motor Non-Multitask Eval Batch

Last updated: 2026-04-06 19:25:23 IST

Queue status: failed
Wait target: `scripts/run_manas_v0p4_nomultitask_batch_20260406.py`
Datasets: `bcic_2a`, `workload`, `motor_mv_img`
Launcher: `torchrun --standalone --nnodes=1 --nproc-per-node=8`

| Run | Status | W&B | Base Config | Run Dir | Trainer Log |
| --- | --- | --- | --- | --- | --- |
| short baseline linear probe + attention pool | failed (1) | [q0aw4rnc](https://wandb.ai/neurodx-labs-neurodx/manas/runs/q0aw4rnc) | `assets/conf/baseline/manas/manas_bench_short_baseline_bcic_workload_motor.yaml` | `/share/arvasunewtemp/eval/manas_short_baseline_lp_attn_20e_bcic_workload_motor_nomultitask_fixed2` | `/share/arvasunewtemp/eval/manas_short_baseline_lp_attn_20e_bcic_workload_motor_nomultitask_fixed2/log/baseline/manas/torchrun_260406192332/manas_trainer.log` |
| short baseline full fine-tune + attention pool | pending | - | `assets/conf/baseline/manas/manas_bench_short_baseline_bcic_workload_motor.yaml` | `/share/arvasunewtemp/eval/manas_short_baseline_fullft_attn_20e_bcic_workload_motor_nomultitask_fixed2` | - |
| long control linear probe + attention pool | pending | - | `assets/conf/baseline/manas/manas_bench_long_control_bcic_workload_motor.yaml` | `/share/arvasunewtemp/eval/manas_long_control_lp_attn_20e_bcic_workload_motor_nomultitask_fixed2` | - |
| long control full fine-tune + attention pool | pending | - | `assets/conf/baseline/manas/manas_bench_long_control_bcic_workload_motor.yaml` | `/share/arvasunewtemp/eval/manas_long_control_fullft_attn_20e_bcic_workload_motor_nomultitask_fixed2` | - |
| v0.4 linear probe + attention pool | pending | - | `assets/conf/baseline/manas/manas_bench_v0p4_parentmemory_bcic_workload_motor.yaml` | `/share/arvasunewtemp/eval/manas_v0p4_parentmemory_lp_attn_20e_bcic_workload_motor_nomultitask_fixed2` | - |
| v0.4 full fine-tune + attention pool | pending | - | `assets/conf/baseline/manas/manas_bench_v0p4_parentmemory_bcic_workload_motor.yaml` | `/share/arvasunewtemp/eval/manas_v0p4_parentmemory_fullft_attn_20e_bcic_workload_motor_nomultitask_fixed2` | - |
