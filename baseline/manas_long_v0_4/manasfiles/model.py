"""
MANAS-Long v0.4 encoder backbone for EEG-FM-Bench.

This mirrors the upstream ndx-pipeline v0.4 flow by loading the exact legacy
parent-memory MAE source that v0.4 revives, then applying the same local-aux
query patch used upstream.
"""

from __future__ import annotations

import subprocess
import types
from pathlib import Path

import torch

_LEGACY_MAE_REPO = Path("~/arvasu/ndx-pipeline").expanduser()
_LEGACY_MAE_COMMIT = "5f3347f"
_LEGACY_MAE_FILE = "models/mae_model.py"
_LEGACY_MAE_MODULE: types.ModuleType | None = None


def _patch_legacy_mae_source(source: str) -> str:
    replacements = [
        (
            """        # A learned query vector to look at the encoder outputs
        self.aux_query = nn.Parameter(torch.randn(1, 1, self.aux_dim))
""",
            """        # Learned readout query/query-set for aux pooling.
        self.num_aux_queries = max(1, self.N_local_aux_tokens)
        self.aux_query = nn.Parameter(torch.randn(1, self.num_aux_queries, self.aux_dim))
""",
        ),
        (
            """            self.memory_token_modules[scale_name] = ScaleMemoryToken(
                embed_dim=self.memory_token_dims[scale],
                bias=use_bias,
            )

    def _init_megatron_transformer_weights(self):
""",
            """            self.memory_token_modules[scale_name] = ScaleMemoryToken(
                embed_dim=self.memory_token_dims[scale],
                bias=use_bias,
            )

    def _aux_query_for_local_idx(self, local_idx: int | None = None) -> torch.Tensor:
        if self.N_local_aux_tokens <= 0 or local_idx is None:
            return self.aux_query[:, :1, :]
        if not 0 <= int(local_idx) < self.N_local_aux_tokens:
            raise IndexError(f"local_idx {local_idx} out of range for {self.N_local_aux_tokens} local aux tokens")
        return self.aux_query[:, int(local_idx) : int(local_idx) + 1, :]

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        aux_query_key = prefix + "aux_query"
        incoming = state_dict.get(aux_query_key)
        if incoming is not None and tuple(incoming.shape) != tuple(self.aux_query.shape):
            if incoming.ndim == 3 and incoming.shape[0] == 1 and incoming.shape[-1] == self.aux_dim:
                if incoming.shape[1] == 1 and self.aux_query.shape[1] > 1:
                    state_dict[aux_query_key] = incoming.expand(-1, self.aux_query.shape[1], -1).clone()
                elif incoming.shape[1] > 1 and self.aux_query.shape[1] == 1:
                    state_dict[aux_query_key] = incoming.mean(dim=1, keepdim=True)
            incoming = state_dict.get(aux_query_key)
            if incoming is not None and tuple(incoming.shape) != tuple(self.aux_query.shape):
                error_msgs.append(
                    f"size mismatch for {aux_query_key}: expected {tuple(self.aux_query.shape)}, got {tuple(incoming.shape)}"
                )
                state_dict.pop(aux_query_key, None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _init_megatron_transformer_weights(self):
""",
        ),
        (
            """        for scale in self.memory_scales_seconds:
            kernel_size = self.memory_conv_kernel_sizes[scale]
            scale_samples = int(round(scale * self.fs))
            if kernel_size > scale_samples:
                raise ValueError(
                    f"memory conv kernel for scale {scale} must be <= scale_samples ({scale_samples}), got {kernel_size}"
                )
""",
            """        if self.use_memory_tokens:
            for scale in self.memory_scales_seconds:
                kernel_size = self.memory_conv_kernel_sizes[scale]
                scale_samples = int(round(scale * self.fs))
                if kernel_size > scale_samples:
                    raise ValueError(
                        f"memory conv kernel for scale {scale} must be <= scale_samples ({scale_samples}), got {kernel_size}"
                    )
""",
        ),
        (
            """        self.memory_patch_embeds = nn.ModuleDict()
        self.memory_token_modules = nn.ModuleDict()
        for scale in self.memory_scales_seconds:
            scale_name = self._memory_scale_name(scale)
            scale_samples = int(round(scale * self.fs))
            scale_step = max(1, int(round(scale_samples * 0.9)))
            self.memory_patch_embeds[scale_name] = RawMemoryPatchEmbed(
                scale_samples=scale_samples,
                scale_step=scale_step,
                conv_kernel_size=self.memory_conv_kernel_sizes[scale],
                base_step=self.step,
                output_dim=self.memory_token_dims[scale],
                conv_channels=self.memory_patch_conv_channels,
                bias=use_bias,
            )
            self.memory_token_modules[scale_name] = ScaleMemoryToken(
                embed_dim=self.memory_token_dims[scale],
                bias=use_bias,
            )
""",
            """        self.memory_patch_embeds = nn.ModuleDict()
        self.memory_token_modules = nn.ModuleDict()
        if self.use_memory_tokens:
            for scale in self.memory_scales_seconds:
                scale_name = self._memory_scale_name(scale)
                scale_samples = int(round(scale * self.fs))
                scale_step = max(1, int(round(scale_samples * 0.9)))
                self.memory_patch_embeds[scale_name] = RawMemoryPatchEmbed(
                    scale_samples=scale_samples,
                    scale_step=scale_step,
                    conv_kernel_size=self.memory_conv_kernel_sizes[scale],
                    base_step=self.step,
                    output_dim=self.memory_token_dims[scale],
                    conv_channels=self.memory_patch_conv_channels,
                    bias=use_bias,
                )
                self.memory_token_modules[scale_name] = ScaleMemoryToken(
                    embed_dim=self.memory_token_dims[scale],
                    bias=use_bias,
                )
""",
        ),
        (
            """            attn_scores = torch.matmul(aux_input, self.aux_query.transpose(1, 2))
""",
            """            attn_scores = torch.matmul(aux_input, self._aux_query_for_local_idx().transpose(1, 2))
""",
        ),
        (
            """            for start, end in self._local_aux_slices(n_vis):
                aux_chunk = aux_input_sorted[:, start:end, :]
                center_chunk = query_centers_vis_sorted[:, start:end]

                chunk_scores = torch.matmul(aux_chunk, self.aux_query.transpose(1, 2))
""",
            """            for local_idx, (start, end) in enumerate(self._local_aux_slices(n_vis)):
                aux_chunk = aux_input_sorted[:, start:end, :]
                center_chunk = query_centers_vis_sorted[:, start:end]

                chunk_query = self._aux_query_for_local_idx(local_idx)
                chunk_scores = torch.matmul(aux_chunk, chunk_query.transpose(1, 2))
""",
        ),
    ]

    patched = source
    for old, new in replacements:
        if old not in patched:
            raise RuntimeError("Failed to patch legacy MAE source for v0.4 local aux queries")
        patched = patched.replace(old, new, 1)
    return patched


def _load_legacy_mae_module() -> types.ModuleType:
    global _LEGACY_MAE_MODULE
    if _LEGACY_MAE_MODULE is not None:
        return _LEGACY_MAE_MODULE

    if not _LEGACY_MAE_REPO.is_dir():
        raise FileNotFoundError(
            f"Expected ndx-pipeline checkout at {_LEGACY_MAE_REPO}, but it does not exist."
        )

    source = subprocess.check_output(
        ["git", "show", f"{_LEGACY_MAE_COMMIT}:{_LEGACY_MAE_FILE}"],
        cwd=str(_LEGACY_MAE_REPO),
        text=True,
    )
    source = _patch_legacy_mae_source(source)

    module = types.ModuleType(f"legacy_mae_{_LEGACY_MAE_COMMIT}")
    exec(compile(source, f"{_LEGACY_MAE_FILE}@{_LEGACY_MAE_COMMIT}", "exec"), module.__dict__)
    _LEGACY_MAE_MODULE = module
    return module


def _get_legacy_mae_class():
    module = _load_legacy_mae_module()
    try:
        return module.MAE
    except AttributeError as exc:
        raise RuntimeError(
            f"Legacy MAE class not found in {_LEGACY_MAE_COMMIT}:{_LEGACY_MAE_FILE}"
        ) from exc


class MAE(_get_legacy_mae_class()):
    """Exact v0.4 MAE backbone plus a downstream classification encode helper."""

    def encode(self, x: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        raw_x = x
        if self.use_memory_tokens and self.use_pairwise_channel_diffs:
            raise NotImplementedError(
                "Raw-signal memory patch generation is not implemented with use_pairwise_channel_diffs=True."
            )

        x, xyz = self._to_pairwise_channels(x, xyz)
        batch_size, num_channels, _ = x.shape

        patches = x.unfold(-1, self.patch_size, self.step)
        num_patches = patches.shape[2]
        tokens = self.patch_embed.linear(patches)
        tokens_flat = tokens.flatten(1, 2)

        coords = self.prepare_coords(xyz, num_patches)
        x_full = tokens_flat + self.pos_enc(coords)

        memory_tokens, memory_centers = self._build_memory_tokens(
            raw_x=raw_x,
            token_mask_grid=None,
            detach_output=False,
        )

        time_idx_full = coords[..., self.n_spatial_coords]
        query_centers_full = time_idx_full * float(self.step) + (0.5 * float(self.patch_size))
        encoder_parent_indices = {
            scale_name: self._memory_parent_indices(query_centers=query_centers_full, memory_centers=centers)
            for scale_name, centers in memory_centers.items()
            if scale_name in memory_tokens
        }

        x_encoded, _, _ = self.encoder(
            x_full,
            memory_tokens=memory_tokens,
            memory_parent_indices=encoder_parent_indices,
        )
        return x_encoded.reshape(batch_size, num_channels, num_patches, self.embed_dim)
