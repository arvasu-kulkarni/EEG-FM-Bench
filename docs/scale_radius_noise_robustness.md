# Scale-Radius Coordinate Robustness Experiment

## Why this experiment was created

This experiment was created to make the model more robust to noise in electrode coordinates while keeping the perturbation physically realistic.

The earlier, simpler way to add coordinate noise is to perturb each electrode position with i.i.d. Gaussian noise in `x`, `y`, and `z`. That is easy to implement, but it is not a very good model of how electrode geometry actually changes in practice. In real EEG recordings, electrode locations usually do not drift independently in arbitrary Cartesian directions. What changes more often is:

- overall head-size mismatch
- cap stretch or compression
- coarse registration error
- mild radial displacement relative to the head center
- small systematic layout mismatch between caps and datasets

Those effects tend to preserve the rough angular arrangement of electrodes on the scalp. A radial scaling perturbation is therefore a more realistic approximation than fully independent Gaussian jitter in each coordinate.

The design goal was:

- improve robustness to coordinate mismatch during pretraining
- preserve spatial topology better than i.i.d. Cartesian noise
- encourage invariance to plausible montage variation, not arbitrary geometric corruption

## Mathematical view

Let the canonical position of electrode `i` be `p_i in R^3`.

Write each position in polar form relative to the head center:

```text
p_i = r_i u_i
r_i = ||p_i||_2
u_i = p_i / ||p_i||_2
```

Here, `r_i` is the radius and `u_i` is the unit direction.

### Baseline: i.i.d. Gaussian coordinate noise

With i.i.d. Gaussian position noise, we perturb each electrode by

```text
p_i' = p_i + eps_i
eps_i ~ N(0, sigma^2 I_3)
```

This changes both radius and direction:

```text
r_i' = ||p_i'||_2
u_i' = p_i' / ||p_i'||_2
```

In general, `u_i' != u_i`.

That means the perturbation can rotate an electrode to a direction that is not anatomically plausible, especially when the coordinate scale is small and `sigma` is not tiny.

### Proposed alternative: radial scaling

Instead of changing the direction, radial scaling keeps the electrode on the same ray from the head center and only changes its radius:

```text
p_i' = s_i p_i
```

where `s_i > 0`.

Two common variants are:

```text
Global scaling:        p_i' = s p_i
Per-electrode scaling: p_i' = s_i p_i
```

If we parameterize the scale as `s_i = 1 + delta_i`, then

```text
p_i' = (1 + delta_i) p_i
```

The key property is:

```text
u_i' = p_i' / ||p_i'||_2 = u_i
r_i' = ||p_i'||_2 = s_i r_i
```

So the angular/topographic structure is preserved while the distance from the center changes.

This is exactly why radial scaling is a better robustness prior for coordinate perturbation:

- it preserves electrode ordering on the scalp
- it preserves the qualitative shape of the montage
- it injects mismatch in a way that is closer to head-size and cap-placement variability

## Why BCIC-IV-2a is the most plausible beneficiary

The strongest gains show up on BCIC-IV-2a, and that is consistent with the geometry of the dataset.

In this benchmark, BCIC-IV-2a is standardized to the following 22-channel layout:

```text
Fz,
FC3, FC1, FCz, FC2, FC4,
C5, C3, C1, Cz, C2, C4, C6,
CP3, CP1, CPz, CP2, CP4,
P1, Pz, P2,
POz
```

This exact layout is defined in [data/dataset/bcic/bcic_2a.py](../data/dataset/bcic/bcic_2a.py). The same file also shows that the raw labels are mapped from nonstandard placeholders (`E2`, `E3`, ..., `E22`) into this 22-channel standardized montage.

The official BCI Competition IV 2a description also states that the dataset uses 22 Ag/AgCl electrodes, with 3.5 cm inter-electrode spacing, left mastoid reference, and right mastoid ground:

- https://www.bbci.de/competition/iv/desc_2a.pdf

Why this matters:

- this is not a broad whole-head clinical 10-20 montage
- it is a narrow motor-imagery-focused fronto-central / centro-parietal subset
- the exact subset is unusual relative to many pretraining montages
- even if individual sites like `C3`, `C4`, or `Cz` are familiar, the full 22-channel geometry is a more out-of-distribution layout

So the hypothesis is:

If pretraining builds robustness to plausible coordinate mismatch, then transfer should improve most when the downstream task uses an unusual or rarely seen electrode geometry. BCIC-IV-2a fits that description well, which is why it is a natural place to expect the biggest benefit.

Put differently, the model may not have seen this exact spatial support during pretraining, so coordinate-robust pretraining helps it treat BCIC-IV-2a as a mild geometric shift rather than as a completely new layout.

## Results

The tables below are transcribed from the provided result screenshots. I assume higher is better. `--` means that setting was not reported in the screenshots.

### BCIC-2A

| Condition | attention_pool + full_ft | attention_pool + linear_probe | attention_pool + partial_ft | flatten_mlp + full_ft | flatten_mlp + linear_probe | flatten_mlp + partial_ft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scale-radius-control | 0.3811 | 0.2552 | 0.3186 | 0.5425 | 0.4957 | 0.4896 |
| scale-radius10 | 0.4644 | 0.2552 | 0.4470 | 0.6181 | 0.5616 | 0.6085 |

Summary:

- mean score improves from `0.4138` to `0.4925`
- mean delta is `+0.0787`
- `5 / 6` reported settings improve
- biggest gains are `+0.1284` for `attention_pool + partial_ft` and `+0.1189` for `flatten_mlp + partial_ft`

### Motor Movement / Imagery

| Condition | attention_pool + full_ft | attention_pool + linear_probe | attention_pool + partial_ft | flatten_mlp + full_ft | flatten_mlp + linear_probe | flatten_mlp + partial_ft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scale-radius-control | 0.5943 | -- | -- | 0.6298 | 0.5599 | 0.5975 |
| scale-radius10 | 0.5838 | -- | -- | 0.6487 | 0.5541 | 0.6009 |

Summary:

- mixed but roughly neutral overall
- mean score changes from `0.5954` to `0.5969`
- mean delta is `+0.0015`

### Workload

| Condition | attention_pool + full_ft | attention_pool + linear_probe | attention_pool + partial_ft | flatten_mlp + full_ft | flatten_mlp + linear_probe | flatten_mlp + partial_ft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scale-radius-control | 0.5925 | 0.5342 | -- | 0.6667 | 0.5958 | 0.5758 |
| scale-radius10 | 0.5975 | 0.5583 | -- | 0.5942 | 0.6283 | 0.6450 |

Summary:

- `4 / 5` reported settings improve
- mean score improves from `0.5930` to `0.6047`
- mean delta is `+0.0117`
- the only regression is `flatten_mlp + full_ft`

### HMC

| Condition | attention_pool + full_ft | attention_pool + linear_probe | attention_pool + partial_ft | flatten_mlp + full_ft | flatten_mlp + linear_probe | flatten_mlp + partial_ft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scale-radius-control | -- | 0.6364 | -- | -- | 0.639327538 | -- |
| scale-radius10 | -- | 0.6278 | -- | -- | 0.652754298 | -- |

Summary:

- mixed but slightly positive overall
- mean score changes from `0.6379` to `0.6403`
- mean delta is `+0.0024`

### Siena Scalp

| Condition | attention_pool + full_ft | attention_pool + linear_probe | attention_pool + partial_ft | flatten_mlp + full_ft | flatten_mlp + linear_probe | flatten_mlp + partial_ft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scale-radius-control | -- | -- | -- | -- | 0.6625 | -- |
| scale-radius10 | -- | -- | -- | -- | 0.7124 | -- |

Summary:

- the reported setting improves by `+0.0499`

## Interpretation

The main pattern is exactly what the experiment was trying to test.

- The benefit is strongest on BCIC-IV-2a, where the downstream montage is the most unusual and likely the most out-of-distribution.
- The gains are still positive on several other tasks, which suggests the augmentation is not merely overfitting to one dataset.
- The mixed results on some tasks are also reasonable: when the downstream montage is already close to what the model has seen before, extra coordinate robustness may matter less.

## Takeaway

The scale-radius experiment was created because radial scaling is a more realistic model of coordinate uncertainty than i.i.d. Gaussian perturbation in Cartesian space.

It preserves anatomical direction, perturbs only radius, and better matches the kinds of geometry mismatch that arise across subjects, caps, and datasets. The results are most convincing on BCIC-IV-2a, which is exactly where we would expect coordinate-robust pretraining to pay off the most.

## Future directions

A few natural extensions could make the coordinate-robustness story stronger:

- **Small rotations.** Apply small global or local rotations to the electrode cloud to simulate mild cap misalignment or head-registration error. Unlike i.i.d. Cartesian noise, rotations preserve pairwise structure while still testing whether the model depends too strongly on exact canonical orientation.
- **Shear transforms.** Apply weak shear transforms to model asymmetric cap deformation or mild nonlinear mismatch between nominal and realized electrode layouts. This would introduce structured geometric distortion without destroying neighborhood relationships.
- **Mixtures of perturbations.** Combine radial scaling with small rotations and weak shear so pretraining sees a broader family of realistic montage shifts rather than a single perturbation mode.
- **Dataset-conditioned perturbation strength.** Tune the perturbation magnitude based on montage density or channel count, since sparse layouts and dense layouts may respond differently to the same geometric augmentation.
- **Explicit OOD-layout evaluation.** Create a targeted benchmark where pretraining and downstream montages are deliberately mismatched, to test whether geometry-aware pretraining consistently improves transfer under unseen channel layouts.
