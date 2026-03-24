# MANAS-Long Integration Notes

## Why this is a separate package

`baseline/manas/` assumes the original MANAS encoder contract:

- a plain patch encoder with no recurrent memory path
- LayerNorm-style transformer blocks
- a downstream wrapper that can expose features by patchifying EEG, adding positional encoding, and running the encoder directly
- checkpoint compatibility with the older MANAS state-dict layout

The long-context model from `~/arvasu/ndx-pipeline` changes that contract in ways that matter for downstream finetuning:

- the encoder is deeper (`22` blocks by default instead of `12`)
- the transformer stack uses `RMSNorm`, custom flash-attention-style `qkv_proj/out_proj`, and `GEGLU` FFNs
- the encoder is memory-augmented, so visible `1s` tokens read from separately generated `5s/10s/30s` memory-token streams
- those memory tokens are built from raw EEG windows with conv projections and GRU updates, so encoder features now depend on an extra preprocessing path, not just patch embeddings
- the upstream training checkpoint is saved from an outer `MAEModel`, so weights usually arrive under a `mae.` prefix and include decoder / auxiliary-loss modules that the downstream bench does not need

Because of that, a new downstream package is cleaner than trying to force these weights into the older `baseline/manas/` classes.

## Downstream contract in EEG-FM-Bench

Adding a new model type in this repo means providing four things:

1. a config class
2. an optional dataset adapter / dataloader factory
3. a trainer that builds the encoder + multi-head classifier
4. a registry entry in `baseline/__init__.py`

For MANAS-family models, the practical downstream encoder contract is:

- input batch fields: `data`, `pos`, `montage`
- output feature shape for the classifier heads: `(B, C, P, E)`
- adapter responsibilities: channel filtering, resampling, z-scoring, and position construction

`manas-long` keeps the same adapter contract as `manas`, then swaps in a different encoder/checkpoint path.

## Forward-path differences that matter here

`manas` downstream flow:

1. resample + normalize EEG
2. build per-channel xyz positions
3. unfold into `1s` patches
4. patch-embed + positional encoding
5. run encoder
6. reshape to `(B, C, P, E)` for the classifier head

`manas-long` downstream flow:

1. resample + normalize EEG
2. build per-channel xyz positions
3. unfold into `1s` patches
4. patch-embed + positional encoding for the visible token grid
5. independently build multi-scale memory tokens from raw EEG windows
6. compute parent indices from each `1s` token center to each memory stream
7. run the memory-augmented encoder
8. reshape to `(B, C, P, E)` for the classifier head

That extra raw-signal memory branch is the main reason a separate setup is required.

## Checkpoint-loading strategy

The upstream long-model checkpoint contains more than the downstream bench needs:

- encoder-side modules that we do need:
  - `patch_embed`
  - `pos_enc`
  - `encoder`
  - `memory_patch_embeds`
  - `memory_token_modules`
- decoder / auxiliary reconstruction modules that are unused in downstream classification

So the `manas-long` trainer strips outer prefixes like `mae.` / `module.` and loads only encoder-side keys into the downstream backbone. This keeps checkpoint compatibility without copying the full pretraining loss stack into EEG-FM-Bench.

## Training-policy differences

`partial_ft` is also slightly different for `manas-long`.

For the original MANAS model, unfreezing only the last encoder block is usually enough because the encoder path is simple. For `manas-long`, the memory pathway is part of the feature computation itself, so `partial_ft` also unfreezes:

- the last transformer block
- the encoder final norm
- the shared memory-branch attention modules
- the memory-token generators

That gives the downstream run a lightweight adaptation path that still reaches the long-context machinery.

## Naming note

The registered model type is `manas-long`, but the Python package is `baseline/manas_long/`.
That split is intentional because Python module names cannot contain `-`.
