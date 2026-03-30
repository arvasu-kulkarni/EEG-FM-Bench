"""
Encoder-side MANAS-Long v0.1 backbone adapted from ndx-pipeline.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from baseline.manas_long.manasfiles.model import (
    PatchEmbed,
    PosEnc,
    ScaleMemoryToken,
    TransformerBlock,
    _memory_scale_name,
)


class ParentMemoryAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        memory_dim: int | None = None,
        attn_dim: int | None = None,
        bias: bool = True,
    ):
        super().__init__()
        self.query_dim = int(query_dim)
        self.memory_dim = self.query_dim if memory_dim is None else int(memory_dim)
        self.attn_dim = self.query_dim if attn_dim is None else int(attn_dim)
        self.head_dim = self.attn_dim
        self.q_proj = nn.Linear(self.query_dim, self.attn_dim, bias=bias)
        self.k_proj = nn.Linear(self.memory_dim, self.attn_dim, bias=bias)
        self.v_proj = nn.Linear(self.memory_dim, self.attn_dim, bias=bias)
        self.out_proj = nn.Linear(self.attn_dim, self.query_dim, bias=bias)

    def forward(
        self,
        query_tokens: torch.Tensor,
        memory_tokens: torch.Tensor,
        parent_indices: torch.Tensor,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query_tokens.ndim != 3 or memory_tokens.ndim != 3:
            raise ValueError("query_tokens and memory_tokens must both have shape (B, T, D)")
        if parent_indices.shape != query_tokens.shape[:2]:
            raise ValueError(
                f"parent_indices shape mismatch: expected {tuple(query_tokens.shape[:2])}, got {tuple(parent_indices.shape)}"
            )

        parent_indices = parent_indices.clamp(min=0, max=max(0, memory_tokens.shape[1] - 1)).long()
        gather_index = parent_indices.unsqueeze(-1).expand(-1, -1, memory_tokens.shape[-1])
        parent_memory = torch.gather(memory_tokens, dim=1, index=gather_index)

        q = self.q_proj(query_tokens).unsqueeze(2)
        k = self.k_proj(parent_memory).unsqueeze(2)
        v = self.v_proj(parent_memory).unsqueeze(2)

        attn_scores = (q * k).sum(dim=-1, keepdim=True) / math.sqrt(float(self.head_dim))
        attn_weights = torch.softmax(attn_scores, dim=2)
        out = (attn_weights * v).sum(dim=2)
        out = self.out_proj(out)

        if query_mask is not None:
            out = out * query_mask.unsqueeze(-1).to(dtype=out.dtype)

        return out


class MemoryAugmentedEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        depth: int = 22,
        heads: int = 8,
        dropout: float = 0.0,
        use_flash_attention: bool = True,
        bias: bool = True,
        memory_scales: tuple[float, ...] = (5.0, 10.0, 30.0),
        memory_token_dims: dict[float, int] | None = None,
        memory_access_start_layers: dict[float, int] | None = None,
    ):
        super().__init__()
        self.memory_scales = tuple(float(scale) for scale in memory_scales)
        self.memory_token_dims = {float(scale): int(dim) for scale, dim in (memory_token_dims or {}).items()}
        self.memory_access_start_layers = {
            float(scale): int(start_layer) for scale, start_layer in (memory_access_start_layers or {}).items()
        }

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    heads=heads,
                    dropout=dropout,
                    use_flash_attention=use_flash_attention,
                    bias=bias,
                )
                for _ in range(depth)
            ]
        )
        self.memory_branches = nn.ModuleDict(
            {
                _memory_scale_name(scale): ParentMemoryAttention(
                    query_dim=embed_dim,
                    memory_dim=self.memory_token_dims.get(scale, embed_dim),
                    bias=bias,
                )
                for scale in self.memory_scales
            }
        )

        from baseline.manas_long.manasfiles.model import RMSNorm

        self.final_norm = RMSNorm(embed_dim)

    def _scales_for_block(self, block_idx_1based: int) -> tuple[float, ...]:
        return tuple(
            scale
            for scale in self.memory_scales
            if block_idx_1based >= self.memory_access_start_layers.get(scale, math.inf)
        )

    def forward(
        self,
        x: torch.Tensor,
        memory_tokens: dict[str, torch.Tensor] | None = None,
        memory_parent_indices: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        attn_intermediates = []
        ffn_intermediates = []

        memory_tokens = memory_tokens or {}
        memory_parent_indices = memory_parent_indices or {}

        for block_idx, layer in enumerate(self.layers, start=1):
            x, attn_out, ffn_out = layer(x)

            for scale in self._scales_for_block(block_idx):
                scale_name = _memory_scale_name(scale)
                scale_memory = memory_tokens.get(scale_name)
                scale_parent = memory_parent_indices.get(scale_name)
                if scale_memory is None or scale_parent is None:
                    continue
                x = x + self.memory_branches[scale_name](
                    query_tokens=x,
                    memory_tokens=scale_memory,
                    parent_indices=scale_parent,
                )

            attn_intermediates.append(attn_out)
            ffn_intermediates.append(ffn_out)

        return self.final_norm(x), attn_intermediates, ffn_intermediates


class RawMemoryPatchEmbed(nn.Module):
    def __init__(
        self,
        scale_samples: int,
        scale_step: int,
        conv_kernel_size: int,
        base_step: int,
        output_dim: int,
        conv_channels: int = 64,
        bias: bool = True,
    ):
        super().__init__()
        if scale_samples <= 0 or scale_step <= 0:
            raise ValueError("scale_samples and scale_step must be > 0")
        if conv_kernel_size <= 0 or base_step <= 0:
            raise ValueError("conv_kernel_size and base_step must be > 0")
        if conv_kernel_size > scale_samples:
            raise ValueError("conv_kernel_size must be <= scale_samples")

        self.scale_samples = int(scale_samples)
        self.scale_step = int(scale_step)
        self.conv_kernel_size = int(conv_kernel_size)
        self.base_step = int(base_step)
        self.conv_channels = int(conv_channels)

        self.tokens_per_window = max(1, ((self.scale_samples - self.conv_kernel_size) // self.base_step) + 1)
        self.start_patch_stride = max(1, self.scale_step // self.base_step)

        self.raw_conv = nn.Conv1d(
            in_channels=1,
            out_channels=self.conv_channels,
            kernel_size=self.conv_kernel_size,
            stride=self.base_step,
            bias=bias,
        )
        self.proj = nn.Linear(self.conv_channels * self.tokens_per_window, output_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        token_mask_grid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        windows = x.unfold(dimension=-1, size=self.scale_samples, step=self.scale_step)
        batch_size, num_channels, num_windows, _ = windows.shape

        conv_in = windows.reshape(batch_size * num_channels * num_windows, 1, self.scale_samples)
        conv_out = self.raw_conv(conv_in).reshape(
            batch_size,
            num_channels,
            num_windows,
            self.conv_channels,
            self.tokens_per_window,
        )

        if token_mask_grid is None:
            channel_weights = torch.ones((batch_size, num_channels, num_windows), device=x.device, dtype=conv_out.dtype)
        else:
            patch_offsets = torch.arange(self.tokens_per_window, device=x.device)
            start_patch_idx = torch.arange(num_windows, device=x.device) * self.start_patch_stride
            patch_idx = (start_patch_idx.unsqueeze(-1) + patch_offsets.unsqueeze(0)).clamp(
                max=token_mask_grid.shape[-1] - 1
            )
            token_mask_expanded = token_mask_grid.unsqueeze(2).expand(-1, -1, num_windows, -1)
            gather_index = patch_idx.view(1, 1, num_windows, self.tokens_per_window).expand(
                batch_size, num_channels, -1, -1
            )
            window_mask = torch.gather(token_mask_expanded, dim=3, index=gather_index).to(dtype=conv_out.dtype)
            conv_out = conv_out * window_mask.unsqueeze(-2)
            channel_weights = window_mask.mean(dim=-1)

        window_embed = self.proj(conv_out.flatten(start_dim=3))
        denom = channel_weights.sum(dim=1, keepdim=False).clamp_min(1.0).unsqueeze(-1)
        pooled = (window_embed * channel_weights.unsqueeze(-1)).sum(dim=1) / denom

        centers = (
            torch.arange(num_windows, device=x.device, dtype=torch.float32) * float(self.scale_step)
            + (0.5 * float(self.scale_samples))
        )
        return pooled, centers


class MAE(nn.Module):
    """Encoder-only subset of the MANAS-Long v0.1 MAE used for downstream classification."""

    def __init__(
        self,
        fs: int = 200,
        patch_seconds: float = 1.0,
        overlap_seconds: float = 0.1,
        embed_dim: int = 512,
        encoder_depth: int = 22,
        encoder_heads: int = 8,
        attn_dropout: float = 0.0,
        use_flash_attention: bool = True,
        use_pairwise_channel_diffs: bool = False,
        n_spatial_coords: int = 3,
        posenc_n_freqs: int = 4,
        remove_bias_except_decoder_out: bool = False,
        use_memory_tokens: bool = True,
        memory_scales_seconds: list[float] | None = None,
        memory_active_scales_seconds: list[float] | None = None,
        memory_attention_scales_seconds: list[float] | None = None,
        memory_encoder_access_start_layers: dict[float | str, int] | None = None,
        memory_conv_kernel_size: int | None = None,
        memory_conv_kernel_sizes: dict[float | str, int] | None = None,
        memory_patch_conv_channels: int = 64,
        memory_token_dim: int | None = None,
        memory_token_dims: dict[float | str, int] | None = None,
    ):
        super().__init__()

        self.embed_dim = int(embed_dim)
        self.fs = int(fs)
        self.use_pairwise_channel_diffs = bool(use_pairwise_channel_diffs)
        self.n_spatial_coords = 6 if self.use_pairwise_channel_diffs else int(n_spatial_coords)
        self.posenc_n_freqs = int(posenc_n_freqs)
        self.attn_dropout = float(attn_dropout)
        self.use_flash_attention = bool(use_flash_attention)
        self.use_memory_tokens = bool(use_memory_tokens)
        self.memory_patch_conv_channels = int(memory_patch_conv_channels)
        self.memory_token_dim = self.embed_dim if memory_token_dim is None else int(memory_token_dim)
        self.memory_conv_kernel_size = None if memory_conv_kernel_size is None else int(memory_conv_kernel_size)

        if self.n_spatial_coords <= 0:
            raise ValueError("n_spatial_coords must be > 0")
        if self.posenc_n_freqs <= 0:
            raise ValueError("posenc_n_freqs must be > 0")
        if not 0.0 <= self.attn_dropout < 1.0:
            raise ValueError("attn_dropout must be in [0, 1)")
        if self.memory_patch_conv_channels <= 0:
            raise ValueError("memory_patch_conv_channels must be > 0")
        if self.memory_token_dim <= 0:
            raise ValueError("memory_token_dim must be > 0")
        if self.memory_conv_kernel_size is not None and self.memory_conv_kernel_size <= 0:
            raise ValueError("memory_conv_kernel_size must be > 0 when provided")

        if memory_scales_seconds is None:
            memory_scales_seconds = [5.0, 10.0, 30.0]
        if memory_active_scales_seconds is None:
            memory_active_scales_seconds = [5.0, 10.0, 30.0]
        if memory_attention_scales_seconds is None:
            memory_attention_scales_seconds = [5.0, 10.0, 30.0]
        if memory_encoder_access_start_layers is None:
            memory_encoder_access_start_layers = {5.0: 9, 10.0: 9, 30.0: 15}
        if memory_conv_kernel_sizes is None:
            memory_conv_kernel_sizes = {}
        if memory_token_dims is None:
            memory_token_dims = {}

        parsed_scales = tuple(float(scale) for scale in memory_scales_seconds)
        parsed_active_scales = tuple(float(scale) for scale in memory_active_scales_seconds)
        parsed_attention_scales = tuple(float(scale) for scale in memory_attention_scales_seconds)

        if any(scale <= 0.0 for scale in parsed_scales):
            raise ValueError("memory_scales_seconds must contain only positive values")
        if any(scale <= 0.0 for scale in parsed_active_scales):
            raise ValueError("memory_active_scales_seconds must contain only positive values")
        if any(scale not in parsed_scales for scale in parsed_active_scales):
            raise ValueError("memory_active_scales_seconds must be a subset of memory_scales_seconds")
        if any(scale <= 0.0 for scale in parsed_attention_scales):
            raise ValueError("memory_attention_scales_seconds must contain only positive values")
        if any(scale not in parsed_scales for scale in parsed_attention_scales):
            raise ValueError("memory_attention_scales_seconds must be a subset of memory_scales_seconds")

        parsed_encoder_access_start_layers = {float(scale): int(start) for scale, start in memory_encoder_access_start_layers.items()}
        parsed_memory_conv_kernel_sizes = {float(scale): int(kernel) for scale, kernel in memory_conv_kernel_sizes.items()}
        parsed_memory_token_dims = {float(scale): int(dim) for scale, dim in memory_token_dims.items()}

        missing_encoder_scales = [scale for scale in parsed_attention_scales if scale not in parsed_encoder_access_start_layers]
        if missing_encoder_scales:
            raise ValueError(
                "memory_encoder_access_start_layers must define a start layer for every attention scale; "
                f"missing {missing_encoder_scales}"
            )

        self.memory_scales_seconds = parsed_scales
        self.memory_active_scales_seconds = parsed_active_scales if self.use_memory_tokens else tuple()
        self.memory_build_order_seconds = tuple(sorted(self.memory_active_scales_seconds))
        self.memory_attention_scales_seconds = parsed_attention_scales if self.use_memory_tokens else tuple()
        self.memory_encoder_access_start_layers = (
            {scale: parsed_encoder_access_start_layers[scale] for scale in self.memory_attention_scales_seconds}
            if self.use_memory_tokens
            else {}
        )

        use_bias = not bool(remove_bias_except_decoder_out)

        self.patch_embed = PatchEmbed(
            fs=fs,
            patch_seconds=patch_seconds,
            overlap_seconds=overlap_seconds,
            embed_dim=embed_dim,
        )
        self.patch_size = self.patch_embed.patch_size
        self.step = self.patch_embed.step

        self.memory_conv_kernel_sizes = (
            {
                scale: int(parsed_memory_conv_kernel_sizes.get(scale, self.memory_conv_kernel_size or self.patch_size))
                for scale in self.memory_scales_seconds
            }
            if self.use_memory_tokens
            else {}
        )
        self.memory_token_dims = (
            {scale: int(parsed_memory_token_dims.get(scale, self.memory_token_dim)) for scale in self.memory_scales_seconds}
            if self.use_memory_tokens
            else {}
        )

        for scale in self.memory_scales_seconds:
            kernel_size = self.memory_conv_kernel_sizes[scale]
            scale_samples = int(round(scale * self.fs))
            if kernel_size > scale_samples:
                raise ValueError(
                    f"memory conv kernel for scale {scale} must be <= scale_samples ({scale_samples}), got {kernel_size}"
                )

        self.pos_enc = PosEnc(
            n_freqs=self.posenc_n_freqs,
            embed_dim=embed_dim,
            n_coords=self.n_spatial_coords + 1,
        )

        self.encoder = MemoryAugmentedEncoder(
            embed_dim=embed_dim,
            depth=encoder_depth,
            heads=encoder_heads,
            dropout=self.attn_dropout,
            use_flash_attention=self.use_flash_attention,
            bias=use_bias,
            memory_scales=self.memory_attention_scales_seconds,
            memory_token_dims=self.memory_token_dims,
            memory_access_start_layers=self.memory_encoder_access_start_layers,
        )

        self.memory_patch_embeds = nn.ModuleDict()
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

    @staticmethod
    def _memory_scale_name(scale_seconds: float) -> str:
        return _memory_scale_name(scale_seconds)

    def prepare_coords(self, xyz: torch.Tensor, num_patches: int):
        batch_size, num_channels, _ = xyz.shape
        if xyz.shape[-1] != self.n_spatial_coords:
            raise ValueError(f"Expected xyz last dim == {self.n_spatial_coords}, got {xyz.shape[-1]}")

        device = xyz.device
        time_idx = torch.arange(num_patches, device=device, dtype=torch.float32)
        spat = xyz.unsqueeze(2).expand(-1, -1, num_patches, -1)
        time = time_idx.view(1, 1, num_patches, 1).expand(batch_size, num_channels, -1, -1)
        coords = torch.cat([spat, time], dim=-1)
        return coords.flatten(1, 2)

    def _to_pairwise_channels(self, x: torch.Tensor, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_pairwise_channel_diffs:
            return x, xyz

        if xyz.shape[-1] != 3:
            raise ValueError(f"Pairwise channel diffs expect xyz with last dim 3, got {xyz.shape[-1]}")
        if x.shape[1] != xyz.shape[1]:
            raise ValueError(f"x/xyz channel mismatch: x has {x.shape[1]}, xyz has {xyz.shape[1]}")
        if x.shape[1] < 2:
            raise ValueError("Pairwise channel diffs require at least 2 channels")

        i, j = torch.triu_indices(x.shape[1], x.shape[1], offset=1, device=x.device)
        x_pair = (x[:, i, :] - x[:, j, :]) / math.sqrt(2.0)
        xyz_pair = torch.cat([xyz[:, i, :], xyz[:, j, :]], dim=-1)

        xyz_orig = torch.cat([xyz, xyz], dim=-1)
        x_all = torch.cat([x, x_pair], dim=1)
        xyz_all = torch.cat([xyz_orig, xyz_pair], dim=1)
        return x_all, xyz_all

    def _build_memory_tokens(
        self,
        raw_x: torch.Tensor,
        token_mask_grid: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.use_memory_tokens or not self.memory_active_scales_seconds:
            return {}, {}

        memory_tokens_by_scale: dict[str, torch.Tensor] = {}
        centers_by_scale: dict[str, torch.Tensor] = {}

        for scale in self.memory_build_order_seconds:
            scale_name = self._memory_scale_name(scale)
            patch_seq, centers = self.memory_patch_embeds[scale_name](
                x=raw_x,
                token_mask_grid=token_mask_grid,
            )
            scale_tokens = self.memory_token_modules[scale_name](sequence=patch_seq)
            memory_tokens_by_scale[scale_name] = scale_tokens
            centers_by_scale[scale_name] = centers

        return memory_tokens_by_scale, centers_by_scale

    @staticmethod
    def _memory_parent_indices(query_centers: torch.Tensor, memory_centers: torch.Tensor) -> torch.Tensor:
        if memory_centers.numel() == 0:
            return torch.zeros_like(query_centers, dtype=torch.long)
        dists = torch.abs(query_centers.unsqueeze(-1) - memory_centers.view(1, 1, -1))
        return torch.argmin(dists, dim=-1)

    def num_patches_for_length(self, total_samples: int) -> int:
        if total_samples < self.patch_size:
            return 0
        return int(((total_samples - self.patch_size) // self.step) + 1)

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

        memory_tokens, memory_centers = self._build_memory_tokens(raw_x=raw_x, token_mask_grid=None)

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

    def forward(self, x: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        return self.encode(x, xyz)
