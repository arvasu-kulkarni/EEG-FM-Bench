"""
Encoder-side MANAS-Long backbone adapted from ndx-pipeline.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_val, gate = x.chunk(2, dim=-1)
        gate_gelu = 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (gate + 0.044715 * gate.pow(3))))
        gate_mix = gate * torch.sigmoid(gate) + gate * gate_gelu
        return x_val * gate_mix


class GEGLUFeedForward(nn.Module):
    def __init__(self, embed_dim: int, expansion: int = 4, bias: bool = True):
        super().__init__()
        hidden_dim = expansion * embed_dim
        self.in_proj = nn.Linear(embed_dim, 2 * hidden_dim, bias=bias)
        self.gate = GEGLU()
        self.out_proj = nn.Linear(hidden_dim, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.gate(self.in_proj(x)))


class PatchEmbed(nn.Module):
    def __init__(self, fs: int = 200, patch_seconds: float = 1.0, overlap_seconds: float = 0.1, embed_dim: int = 512):
        super().__init__()
        self.patch_size = int(round(patch_seconds * fs))
        self.overlap_size = int(round(overlap_seconds * fs))
        self.step = self.patch_size - self.overlap_size
        self.linear = nn.Linear(self.patch_size, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.step)
        return self.linear(patches)


class PosEnc(nn.Module):
    def __init__(self, n_freqs: int = 4, embed_dim: int = 512, n_coords: int = 4):
        super().__init__()
        if n_coords <= 0:
            raise ValueError("n_coords must be > 0")

        self.n_coords = int(n_coords)
        freqs = torch.linspace(1.0, 10.0, n_freqs)
        self.register_buffer("freq_matrix", torch.cartesian_prod(*([freqs] * self.n_coords)).transpose(1, 0))

        fourier_features_dim = 2 * (n_freqs ** self.n_coords)
        self.fourier_linear = nn.Linear(fourier_features_dim, embed_dim, bias=False)
        self.learned_linear = nn.Sequential(
            nn.Linear(self.n_coords, 2 * embed_dim, bias=False),
            GEGLU(),
            RMSNorm(embed_dim),
        )
        self.final_norm = RMSNorm(embed_dim)

    def forward(self, coords: torch.Tensor):
        phases = torch.matmul(coords, self.freq_matrix)
        fourier_features = torch.cat([torch.sin(phases), torch.cos(phases)], -1)
        fourier_emb = self.fourier_linear(fourier_features)
        learned_emb = self.learned_linear(coords)
        return self.final_norm(fourier_emb + learned_emb)


class FlashSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, heads: int, dropout: float = 0.0, use_flash_attention: bool = True, bias: bool = True):
        super().__init__()
        if embed_dim % heads != 0:
            raise ValueError("embed_dim must be divisible by heads")

        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.dropout = float(dropout)
        self.use_sdpa = bool(use_flash_attention) and hasattr(F, "scaled_dot_product_attention")

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)

        if self.use_sdpa:
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            scale = 1.0 / math.sqrt(float(self.head_dim))
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            if self.dropout > 0.0 and self.training:
                attn_weights = F.dropout(attn_weights, p=self.dropout)
            attn_out = torch.matmul(attn_weights, v)

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, heads: int, dropout: float = 0.0, use_flash_attention: bool = True, bias: bool = True):
        super().__init__()
        self.pre_attn_norm = RMSNorm(embed_dim)
        self.attn = FlashSelfAttention(
            embed_dim=embed_dim,
            heads=heads,
            dropout=dropout,
            use_flash_attention=use_flash_attention,
            bias=bias,
        )
        self.pre_ffn_norm = RMSNorm(embed_dim)
        self.ffn = GEGLUFeedForward(embed_dim=embed_dim, expansion=4, bias=bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_in = self.pre_attn_norm(x)
        attn_out = self.attn(attn_in)
        x = x + attn_out

        ffn_in = self.pre_ffn_norm(x)
        ffn_out = self.ffn(ffn_in)
        x = x + ffn_out
        return x, attn_out, ffn_out


def _memory_scale_name(scale_seconds: float) -> str:
    return f"scale_{str(scale_seconds).replace('.', 'p')}"


class ParentMemoryAttention(nn.Module):
    def __init__(self, embed_dim: int, bias: bool = True):
        super().__init__()
        self.head_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

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
    ):
        super().__init__()
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
                _memory_scale_name(scale): ParentMemoryAttention(embed_dim=embed_dim, bias=bias)
                for scale in memory_scales
            }
        )
        self.final_norm = RMSNorm(embed_dim)

    @staticmethod
    def _scales_for_block(block_idx_1based: int) -> tuple[float, ...]:
        if block_idx_1based <= 8:
            return tuple()
        if block_idx_1based <= 14:
            return (5.0, 10.0)
        return (5.0, 10.0, 30.0)

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
        base_patch_size: int,
        base_step: int,
        embed_dim: int,
        conv_channels: int = 64,
        bias: bool = True,
    ):
        super().__init__()
        if scale_samples <= 0 or scale_step <= 0:
            raise ValueError("scale_samples and scale_step must be > 0")
        if base_patch_size <= 0 or base_step <= 0:
            raise ValueError("base_patch_size and base_step must be > 0")

        self.scale_samples = int(scale_samples)
        self.scale_step = int(scale_step)
        self.base_patch_size = int(base_patch_size)
        self.base_step = int(base_step)
        self.conv_channels = int(conv_channels)

        self.tokens_per_window = max(1, ((self.scale_samples - self.base_patch_size) // self.base_step) + 1)
        self.start_patch_stride = max(1, self.scale_step // self.base_step)

        self.raw_conv = nn.Conv1d(
            in_channels=1,
            out_channels=self.conv_channels,
            kernel_size=self.base_patch_size,
            stride=self.base_step,
            bias=bias,
        )
        self.proj = nn.Linear(self.conv_channels * self.tokens_per_window, embed_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        token_mask_grid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-1] < self.scale_samples:
            x = F.pad(x, (0, self.scale_samples - x.shape[-1]))

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
            channel_weights = torch.ones(
                (batch_size, num_channels, num_windows),
                device=x.device,
                dtype=conv_out.dtype,
            )
        else:
            patch_offsets = torch.arange(self.tokens_per_window, device=x.device)
            start_patch_idx = torch.arange(num_windows, device=x.device) * self.start_patch_stride
            patch_idx = (start_patch_idx.unsqueeze(-1) + patch_offsets.unsqueeze(0)).clamp(
                max=token_mask_grid.shape[-1] - 1
            )
            token_mask_expanded = token_mask_grid.unsqueeze(2).expand(-1, -1, num_windows, -1)
            gather_index = patch_idx.view(1, 1, num_windows, self.tokens_per_window).expand(
                batch_size,
                num_channels,
                -1,
                -1,
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


class ScaleMemoryToken(nn.Module):
    def __init__(self, embed_dim: int, bias: bool = True):
        super().__init__()
        self.state_norm = RMSNorm(embed_dim)
        self.new_info_norm = RMSNorm(embed_dim)
        self.gru = nn.GRUCell(embed_dim, embed_dim, bias=bias)
        self.init_state = nn.Parameter(torch.zeros(1, embed_dim))

    def forward(
        self,
        sequence: torch.Tensor,
        time_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sequence.ndim != 3:
            raise ValueError(f"Expected sequence shape (B, P, D), got {tuple(sequence.shape)}")

        batch_size, num_steps, _ = sequence.shape
        if num_steps == 0:
            return sequence.new_zeros((batch_size, 0, sequence.shape[-1]))

        state = self.init_state.expand(batch_size, -1)
        outputs = []
        weights = None if time_weights is None else time_weights.to(dtype=sequence.dtype)

        for step_idx in range(num_steps):
            new_info = sequence[:, step_idx, :]
            if weights is not None:
                new_info = new_info * weights[:, step_idx].unsqueeze(-1)
            new_info = self.new_info_norm(new_info)
            state = self.gru(new_info, state)
            outputs.append(self.state_norm(state).unsqueeze(1))

        return torch.cat(outputs, dim=1)


class MAE(nn.Module):
    """Encoder-only subset of the ndx-pipeline MAE used for downstream classification."""

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
        memory_conv_kernel_size: int = 3,
        memory_patch_conv_channels: int = 64,
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
        self.memory_conv_kernel_size = int(memory_conv_kernel_size)
        self.memory_patch_conv_channels = int(memory_patch_conv_channels)

        if self.n_spatial_coords <= 0:
            raise ValueError("n_spatial_coords must be > 0")
        if self.posenc_n_freqs <= 0:
            raise ValueError("posenc_n_freqs must be > 0")
        if not 0.0 <= self.attn_dropout < 1.0:
            raise ValueError("attn_dropout must be in [0, 1)")
        if self.memory_patch_conv_channels <= 0:
            raise ValueError("memory_patch_conv_channels must be > 0")
        if self.memory_conv_kernel_size <= 0 or self.memory_conv_kernel_size % 2 == 0:
            raise ValueError("memory_conv_kernel_size must be a positive odd integer")

        if memory_scales_seconds is None:
            memory_scales_seconds = [5.0, 10.0, 30.0]
        if memory_active_scales_seconds is None:
            memory_active_scales_seconds = [5.0, 10.0, 30.0]
        if memory_attention_scales_seconds is None:
            memory_attention_scales_seconds = [5.0, 10.0, 30.0]

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

        self.memory_scales_seconds = parsed_scales
        self.memory_active_scales_seconds = parsed_active_scales if self.use_memory_tokens else tuple()
        self.memory_build_order_seconds = tuple(sorted(self.memory_active_scales_seconds))
        self.memory_attention_scales_seconds = parsed_attention_scales if self.use_memory_tokens else tuple()

        use_bias = not bool(remove_bias_except_decoder_out)

        self.patch_embed = PatchEmbed(
            fs=fs,
            patch_seconds=patch_seconds,
            overlap_seconds=overlap_seconds,
            embed_dim=embed_dim,
        )
        self.patch_size = self.patch_embed.patch_size
        self.step = self.patch_embed.step

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
                base_patch_size=self.patch_size,
                base_step=self.step,
                embed_dim=embed_dim,
                conv_channels=self.memory_patch_conv_channels,
                bias=use_bias,
            )
            self.memory_token_modules[scale_name] = ScaleMemoryToken(
                embed_dim=embed_dim,
                bias=use_bias,
            )

    @staticmethod
    def _memory_scale_name(scale_seconds: float) -> str:
        return _memory_scale_name(scale_seconds)

    def prepare_coords(self, xyz: torch.Tensor, num_patches: int):
        batch_size, num_channels, _ = xyz.shape
        if xyz.shape[-1] != self.n_spatial_coords:
            raise ValueError(
                f"Expected xyz last dim == {self.n_spatial_coords}, got {xyz.shape[-1]}"
            )

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
            raise ValueError(
                f"Pairwise channel diffs expect xyz with last dim 3, got {xyz.shape[-1]}"
            )
        if x.shape[1] != xyz.shape[1]:
            raise ValueError(
                f"x/xyz channel mismatch: x has {x.shape[1]}, xyz has {xyz.shape[1]}"
            )
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
    def _memory_parent_indices(
        query_centers: torch.Tensor,
        memory_centers: torch.Tensor,
    ) -> torch.Tensor:
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
            scale_name: self._memory_parent_indices(
                query_centers=query_centers_full,
                memory_centers=centers,
            )
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
