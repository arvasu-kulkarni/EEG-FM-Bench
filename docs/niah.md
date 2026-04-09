# EEG NIAH Objective Ideas

## Goal

This note captures an initial set of EEG-flavored needle-in-a-haystack objectives.

The main design principle is:

- take a long EEG context window that is mostly "normal" or internally consistent
- insert a relatively small synthetic or borrowed needle event
- ask the model to either localize the needle or classify whether the context was modified

These are not meant to be the final benchmark definitions. They are intended as a compact v0 design space that can be implemented mostly from existing cached windows in this repo.

## A. Localization Objectives

### v0: Global synthetic artifact localization

#### Summary

Start with a long EEG window and inject a synthetic high-amplitude spike-like artifact across all channels. The model is asked to predict the temporal center and duration of the inserted event.

#### Input

- a long EEG window `x in R^(C x T)`
- source window should ideally be a relatively clean or homogeneous segment
- one synthetic artifact is inserted at a random time

#### Needle

- a simple high-amplitude transient
- same temporal waveform on all channels
- random sign, amplitude, width, and location
- initially global across channels to keep the task purely temporal

#### Target

- temporal center `t_center`
- temporal duration `d`

A practical implementation is to regress normalized values:

- `t_center / T`
- `d / T`

#### Why this is a good v0

- very easy to generate
- easy to verify visually
- gives a clean localization objective before introducing spatial complexity
- acts as a direct test of whether long-context models can recover a brief event from a long background

#### Suggested metrics

- center absolute error
- duration absolute error
- temporal IoU after converting `(center, duration)` into a span
- hit@tau: whether predicted center falls within a tolerance window

## B. Joint Temporal-Spatial Localization

### v0.1: Spatially localized synthetic artifact

#### Summary

Move from a purely temporal needle to a localized spatiotemporal needle. Instead of affecting all channels, the injected artifact is centered at a spatial location and decays with spatial radius.

#### Input

- a long EEG window `x in R^(C x T)`
- one synthetic artifact inserted at a random time
- artifact centered at a chosen electrode or scalp coordinate

#### Needle

- temporal transient as in `v0`
- spatial support controlled by a center and radius
- two simple versions are possible:
  - hard radius: all channels within radius are affected equally
  - soft radius: channel amplitude decays with distance from spatial center

#### Target

- temporal center
- temporal radius or duration
- spatial center
- spatial radius

One simple parameterization is:

- temporal center `t_center`
- temporal radius `r_t`
- spatial center `s_center`
- spatial radius `r_s`

where `s_center` can be represented either as:

- a discrete center electrode index
- a continuous scalp coordinate if you want to use electrode positions directly

#### Why this is a useful next step

- adds spatial reasoning without requiring dense segmentation
- aligns naturally with EEG montages and the position-aware MANAS family
- gives a more realistic notion of "where the needle is" than a global artifact

#### Suggested metrics

- temporal center error
- temporal radius or duration error
- spatial center accuracy if discrete
- spatial center distance if continuous
- spatial radius error
- spatiotemporal IoU if the prediction is converted into a time span plus channel set

#### Note

A literal 2D image-style bounding box is possible, but EEG may be better served by:

- temporal center plus temporal radius
- spatial center plus spatial radius

or by converting the spatial prediction into a channel mask during evaluation.

## C. Classification Variants

### c-types

The `c` versions reuse the same basic data generation idea but convert the task into classification rather than localization.

### c0: Artifact presence detection

#### Summary

Use the same setup as `v0`, but mix positive and negative examples. Positive examples contain one inserted artifact. Negative examples are left unchanged. The model predicts whether an artifact is present.

#### Input

- same long-window construction as `v0`
- some windows are modified
- some windows are untouched

#### Target

- binary label: artifact present vs artifact absent

#### Why this is useful

- easiest possible NIAH classification baseline
- useful for quick model comparisons before regression or localization
- lets us test whether the model notices the needle at all, even if it cannot localize it precisely

#### Suggested metrics

- AUROC
- AUPRC
- accuracy
- balanced accuracy

### c0.1: Class contamination detection in otherwise homogeneous windows

#### Summary

Take a long window that should belong to a single coherent class, then mix in a short segment from a different class. Instead of asking for localization, ask the model to classify some property of the contamination.

The motivating example is:

- start from a homogeneous `adftd_30` or `adftd_90` window from one class
- replace or mix in a shorter segment, such as `5s`, from another class

#### Core question

This idea is good, but the exact classification target needs to be chosen carefully.

The most plausible target definitions are:

- `c0.1a`: purity detection
  - label = whether the window is pure or contaminated
- `c0.1b`: foreign-class identification
  - label = which class was inserted
- `c0.1c`: base-class recovery under contamination
  - label = original class of the host window despite the inserted foreign segment

#### Recommendation

For a first pass, `c0.1a` is the cleanest objective:

- input = mostly single-class window, optionally with a short foreign segment inserted
- label = pure vs contaminated

This keeps the task conceptually close to NIAH:

- the window is mostly haystack
- the inserted foreign segment is the needle
- the model only has to decide whether the needle exists

After that, `c0.1b` is a natural harder version:

- if contaminated, classify which foreign class was inserted

#### Why this family is interesting

- uses existing class-labeled windows
- does not require hand-designed synthetic waveforms
- tests whether the model notices brief semantic inconsistency inside otherwise coherent long context
- may be a better bridge between classic downstream classification and true NIAH evaluation

#### Suggested metrics

- for `c0.1a`: AUROC, AUPRC, balanced accuracy
- for `c0.1b`: macro-F1, balanced accuracy, confusion matrix
- for `c0.1c`: macro-F1 and robustness relative to the uncontaminated baseline

## Practical ordering

If we want a staged roadmap, the most sensible order is:

1. `v0`: global synthetic artifact, predict temporal center plus duration
2. `c0`: same data generation as `v0`, but binary presence detection
3. `v0.1`: spatially localized synthetic artifact, predict temporal and spatial center plus radius
4. `c0.1`: class-contamination detection or foreign-class identification

This ordering gives:

- one simple regression objective
- one simple classification objective
- one richer spatiotemporal objective
- one semantically meaningful classification objective based on real class structure

## Open choices

A few details are still intentionally undecided:

- whether the inserted artifact should be purely synthetic or copied from real artifact snippets
- whether spatial support should be hard-radius or soft-radius
- whether `c0.1` should be framed as contamination detection, foreign-class identification, or host-class recovery
- whether evaluation should be done on raw seconds or normalized coordinates

Those choices can be resolved later without changing the high-level structure above.
