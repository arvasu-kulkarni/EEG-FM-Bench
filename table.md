# Metrics by Dataset

Rows are scale conditions. Columns are `train_method + head` combinations. Bold marks the better value only when the absolute difference is at least `0.02`.

## BCIC-2A

| Condition | attention_pool + full_ft | attention_pool + linear_probe | attention_pool + partial_ft | flatten_mlp + full_ft | flatten_mlp + linear_probe | flatten_mlp + partial_ft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scale-radius-control | 0.3811 | 0.2552 | 0.3186 | 0.5425 | 0.4957 | 0.4896 |
| scale-radius10 | **0.4644** | 0.2552 | **0.4470** | **0.6181** | **0.5616** | **0.6085** |

## Motor Movement / Imagery

| Condition | attention_pool + full_ft | attention_pool + linear_probe | attention_pool + partial_ft | flatten_mlp + full_ft | flatten_mlp + linear_probe | flatten_mlp + partial_ft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scale-radius-control | 0.5943 | -- | -- | 0.6298 | 0.5599 | 0.5975 |
| scale-radius10 | 0.5838 | -- | -- | 0.6487 | 0.5541 | 0.6009 |

## Workload

| Condition | attention_pool + full_ft | attention_pool + linear_probe | attention_pool + partial_ft | flatten_mlp + full_ft | flatten_mlp + linear_probe | flatten_mlp + partial_ft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scale-radius-control | 0.5925 | 0.5342 | -- | **0.6667** | 0.5958 | 0.5758 |
| scale-radius10 | 0.5975 | **0.5583** | -- | 0.5942 | **0.6283** | **0.6450** |

## HMC

| Condition | attention_pool + full_ft | attention_pool + linear_probe | attention_pool + partial_ft | flatten_mlp + full_ft | flatten_mlp + linear_probe | flatten_mlp + partial_ft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scale-radius-control | -- | 0.6364 | -- | -- | 0.639327538 | -- |
| scale-radius10 | -- | 0.6278 | -- | -- | 0.652754298 | -- |

## Siena Scalp

| Condition | attention_pool + full_ft | attention_pool + linear_probe | attention_pool + partial_ft | flatten_mlp + full_ft | flatten_mlp + linear_probe | flatten_mlp + partial_ft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scale-radius-control | -- | -- | -- | -- | 0.6625 | -- |
| scale-radius10 | -- | -- | -- | -- | **0.7124** | -- |
