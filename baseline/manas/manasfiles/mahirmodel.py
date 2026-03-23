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

        fourier_features_dim = 2 * (n_freqs**self.n_coords)

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


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, heads: int, dropout: float = 0.0):
        super().__init__()

        assert embed_dim % heads == 0, "dim must be divisible by heads"

        self.pre_attn_norm = RMSNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads, dropout=dropout, batch_first=True)

        self.pre_ffn_norm = RMSNorm(embed_dim)
        self.ffn = GEGLUFeedForward(embed_dim=embed_dim, expansion=4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_in = self.pre_attn_norm(x)

        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        x = x + attn_out

        ffn_in = self.pre_ffn_norm(x)

        ffn_out = self.ffn(ffn_in)
        x = x + ffn_out

        return x, ffn_out


class TransformerEncoderDecoder(nn.Module):
    def __init__(self, embed_dim: int = 512, depth: int = 16, heads: int = 8):
        super().__init__()

        self.layers = nn.ModuleList([TransformerBlock(embed_dim, heads) for _ in range(depth)])
        self.final_norm = RMSNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_intermediate: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        intermediate = [] if return_intermediate else None

        for layer in self.layers:
            x, ffn_out = layer(x)
            if return_intermediate:
                intermediate.append(ffn_out)

        return self.final_norm(x), intermediate


class MAEDecoder(nn.Module):
    def __init__(self, embed_dim: int = 512, decoder_depth: int = 4, decoder_heads: int = 8, patch_size: int = 200):
        super().__init__()

        # 1. The Mask Token (The "Gray Tile")
        # A learnable vector that replaces every missing patch
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # 2. The Decoder Transformer (Reuse your Encoder logic)
        # It's lighter (fewer layers) than the main Encoder
        self.decoder = TransformerEncoderDecoder(embed_dim=embed_dim, depth=decoder_depth, heads=decoder_heads)

        # 3. The Prediction Head
        # Projects Vector (512) -> Raw Signal (200)
        self.predict = nn.Linear(embed_dim, patch_size, bias=True)

    def forward(self, x_visible: torch.Tensor, pos_enc: nn.Module, coords: torch.Tensor, mask: torch.Tensor):
        B, N_Total, D = coords.shape[0], coords.shape[1], x_visible.shape[-1]

        # --- Step A: Fill Canvas with Mask Tokens ---
        # Create a tensor of size (Batch, Total, Dim) filled with the mask token
        x_full = self.mask_token.expand(B, N_Total, D).clone()

        # --- Step B: Paste Visible Tokens ---
        # Visible token count is fixed per sample, so boolean flatten assignment is valid.
        x_full[mask] = x_visible.reshape(-1, D)

        # --- Step C: Add Positional Encoding ---
        # We call YOUR PosEnc class here.
        # It takes coords (B, N_Total, n_coords) and returns (B, N_Total, Dim)
        pos_emb = pos_enc(coords)

        # Add GPS info to the tokens
        x_full = x_full + pos_emb

        # --- Step D: Decode ---
        # Pass through the Transformer
        # We ignore the intermediate outputs (the second return value) for now
        x_decoded, _ = self.decoder(x_full, return_intermediate=False)

        # --- Step E: Predict ---
        # (Batch, N_Total, 512) -> (Batch, N_Total, 200)
        prediction = self.predict(x_decoded)

        return prediction


def _split_spatiotemporal_coords(coords: torch.Tensor, n_spatial_coords: int) -> tuple[torch.Tensor, torch.Tensor]:
    if n_spatial_coords <= 0:
        raise ValueError("n_spatial_coords must be > 0")
    if coords.shape[-1] <= n_spatial_coords:
        raise ValueError(
            f"Expected coords last dim > n_spatial_coords, got {coords.shape[-1]} and {n_spatial_coords}"
        )
    spatial = coords[:, :, :n_spatial_coords]
    temporal = coords[:, :, n_spatial_coords]
    return spatial, temporal


def generate_mask(
    coords: torch.Tensor,
    mask_ratio: float = 0.55,
    spatial_radius: float = 3.0,
    temporal_radius: float = 3.0,
    n_spatial_coords: int = 3,
):
    B, N, _ = coords.shape
    device = coords.device
    num_masked_target = int(mask_ratio * N)

    spatial, temporal = _split_spatiotemporal_coords(coords, n_spatial_coords=n_spatial_coords)
    dists_spatial = torch.cdist(spatial, spatial)
    dists_temporal = torch.abs(temporal.unsqueeze(2) - temporal.unsqueeze(1))
    block_black = (dists_spatial <= spatial_radius) & (dists_temporal <= temporal_radius)

    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    masked_count = torch.zeros(B, dtype=torch.long, device=device)
    row_idx = torch.arange(B, device=device)
    max_iters = max(8, N * 4)

    for _ in range(max_iters):
        active = masked_count < num_masked_target
        if not torch.any(active):
            break
        seed_idx = torch.randint(0, N, (B,), device=device)
        chosen = block_black[row_idx, seed_idx] & active.unsqueeze(1)
        new_masked = chosen & mask
        mask = mask & (~chosen)
        masked_count = masked_count + new_masked.sum(dim=1)

    # Enforce exact masked count per sample (fix over/under-shoot from stochastic blocks).
    rand = torch.rand(B, N, device=device)
    masked_scores = torch.where(~mask, rand, rand + 2.0)
    keep_masked_idx = torch.argsort(masked_scores, dim=1)[:, :num_masked_target]
    final_masked = torch.zeros(B, N, dtype=torch.bool, device=device)
    final_masked.scatter_(1, keep_masked_idx, True)

    return ~final_masked


def generate_partial_mask(
    coords: torch.Tensor,
    num_channels: int,
    num_patches: int,
    mask_ratio: float = 0.55,
    spatial_radius_black: float = 3.0,
    spatial_radius_fuzzy: float = 6.0,
    temporal_radius_black: float = 3.0,
    temporal_radius_fuzzy: float = 6.0,
    dropout_ratio: float = 0.0,
    dropout_radius: float = 3.0,
    n_spatial_coords: int = 3,
):
    B, N, _ = coords.shape
    device = coords.device
    if N != num_channels * num_patches:
        raise ValueError(
            f"Expected N == C*P, got N={N}, C={num_channels}, P={num_patches}."
        )

    # Calculate exact number of tokens to hide.
    num_masked_target = int(mask_ratio * N)
    if num_masked_target <= 0:
        mask = torch.ones(B, N, dtype=torch.bool, device=device)
        return mask, torch.zeros_like(mask)

    spatial, temporal = _split_spatiotemporal_coords(coords, n_spatial_coords=n_spatial_coords)

    token_spatial_dists = torch.cdist(spatial, spatial)
    token_temporal_dists = (temporal.unsqueeze(2) - temporal.unsqueeze(1)).abs()
    token_black_neighbors = (
        (token_spatial_dists <= spatial_radius_black) & (token_temporal_dists <= temporal_radius_black)
    )
    token_fuzzy_neighbors = (
        (token_spatial_dists <= spatial_radius_fuzzy) & (token_temporal_dists <= temporal_radius_fuzzy)
    )

    # Seed a block mask in a vectorized way, then exact-correct to target count.
    avg_black_block = max(1.0, float(token_black_neighbors.float().sum(dim=2).mean().item()))
    num_seeds = min(N, max(1, int(math.ceil(num_masked_target / avg_black_block))))
    seed_scores = torch.rand((B, N), device=device)
    seed_idx = seed_scores.topk(k=num_seeds, dim=1, largest=False).indices
    seed_mask = torch.zeros((B, N), dtype=torch.bool, device=device)
    seed_mask.scatter_(1, seed_idx, True)
    block_mask = (token_black_neighbors & seed_mask.unsqueeze(1)).any(dim=2)

    dropped_token_mask = torch.zeros((B, N), dtype=torch.bool, device=device)
    drop_channel_target = min(
        num_channels,
        int((dropout_ratio * num_masked_target) / max(1, num_patches)),
    )
    if drop_channel_target > 0:
        channel_spatial = spatial.view(B, num_channels, num_patches, n_spatial_coords)[:, :, 0, :]
        channel_dists = torch.cdist(channel_spatial, channel_spatial)
        channel_neighbors = channel_dists <= dropout_radius

        seed_scores_ch = torch.rand((B, num_channels), device=device)
        seed_idx_ch = seed_scores_ch.topk(k=drop_channel_target, dim=1, largest=False).indices
        seed_mask_ch = torch.zeros((B, num_channels), dtype=torch.bool, device=device)
        seed_mask_ch.scatter_(1, seed_idx_ch, True)
        dropped_channels = (channel_neighbors & seed_mask_ch.unsqueeze(1)).any(dim=2)

        # Keep exact channel-drop budget.
        curr_drop = dropped_channels.sum(dim=1)
        over = (curr_drop - drop_channel_target).clamp_min(0)
        if int(over.max().item()) > 0:
            keep_scores = torch.rand((B, num_channels), device=device)
            keep_scores = keep_scores.masked_fill(~dropped_channels, 2.0)
            max_over = int(over.max().item())
            idx = keep_scores.topk(k=max_over, dim=1, largest=False).indices
            choose = torch.arange(max_over, device=device).view(1, -1) < over.view(-1, 1)
            b_idx = torch.arange(B, device=device).view(-1, 1).expand(-1, max_over)
            dropped_channels[b_idx[choose], idx[choose]] = False

        channel_ids = (
            torch.arange(num_channels, device=device)
            .view(1, num_channels, 1)
            .expand(B, -1, num_patches)
            .reshape(B, -1)
        )
        dropped_token_mask = torch.gather(dropped_channels, 1, channel_ids)

    # mask=True means visible token, mask=False means fully hidden token.
    mask = ~(block_mask | dropped_token_mask)

    # Exact-count correction without undoing mandatory dropped channels.
    curr_masked = (~mask).sum(dim=1)
    over = (curr_masked - num_masked_target).clamp_min(0)
    max_over = int(over.max().item())
    if max_over > 0:
        can_unmask = (~mask) & (~dropped_token_mask)
        scores = torch.rand((B, N), device=device).masked_fill(~can_unmask, 2.0)
        idx = scores.topk(k=max_over, dim=1, largest=False).indices
        choose = torch.arange(max_over, device=device).view(1, -1) < over.view(-1, 1)
        b_idx = torch.arange(B, device=device).view(-1, 1).expand(-1, max_over)
        mask[b_idx[choose], idx[choose]] = True

    curr_masked = (~mask).sum(dim=1)
    need = (num_masked_target - curr_masked).clamp_min(0)
    max_need = int(need.max().item())
    if max_need > 0:
        can_mask = mask & (~dropped_token_mask)
        scores = torch.rand((B, N), device=device).masked_fill(~can_mask, 2.0)
        idx = scores.topk(k=max_need, dim=1, largest=False).indices
        choose = torch.arange(max_need, device=device).view(1, -1) < need.view(-1, 1)
        b_idx = torch.arange(B, device=device).view(-1, 1).expand(-1, max_need)
        mask[b_idx[choose], idx[choose]] = False

    mask_fz = (token_fuzzy_neighbors & (~mask).unsqueeze(1)).any(dim=2) & mask

    return mask, mask_fz


class MAE(nn.Module):
    def __init__(
        self,
        # Data Params
        fs: int = 200,
        patch_seconds: float = 1.0,
        overlap_seconds: float = 0.1,
        # Model Params
        embed_dim: int = 512,
        encoder_depth: int = 12,
        encoder_heads: int = 8,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
        # Training Params
        mask_ratio: float = 0.55,
        aux_loss_weight: float = 0.1,
        which_mask: str = "default",
        fuzzy_noise_std: float = 0.1,
        spatial_radius_black: float = 3.0,
        spatial_radius_fuzzy: float = 6.0,
        temporal_radius_black: float = 3.0,
        temporal_radius_fuzzy: float = 6.0,
        dropout_ratio: float = 0.0,
        dropout_radius: float = 3.0,
        ema_mix_ratio: float = 0.6,
        ema_temperature: float = 2.0,
        ema_floor_eps: float = 0.1,
        use_pairwise_channel_diffs: bool = False,
        n_spatial_coords: int = 3,
        posenc_n_freqs: int = 4,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.aux_loss_weight = aux_loss_weight
        self.which_mask = which_mask
        self.fuzzy_noise_std = fuzzy_noise_std
        self.spatial_radius_black = spatial_radius_black
        self.spatial_radius_fuzzy = spatial_radius_fuzzy
        self.temporal_radius_black = temporal_radius_black
        self.temporal_radius_fuzzy = temporal_radius_fuzzy
        self.dropout_ratio = dropout_ratio
        self.dropout_radius = dropout_radius
        self.ema_mix_ratio = ema_mix_ratio
        self.ema_temperature = ema_temperature
        self.ema_floor_eps = ema_floor_eps
        self.use_pairwise_channel_diffs = bool(use_pairwise_channel_diffs)
        self.n_spatial_coords = 6 if self.use_pairwise_channel_diffs else int(n_spatial_coords)
        self.posenc_n_freqs = int(posenc_n_freqs)

        if self.which_mask not in {"default", "fuzzy"}:
            raise ValueError(f"which_mask must be one of ['default', 'fuzzy'], got '{self.which_mask}'")
        if self.fuzzy_noise_std < 0.0:
            raise ValueError("fuzzy_noise_std must be >= 0")
        if self.spatial_radius_fuzzy < self.spatial_radius_black:
            raise ValueError("spatial_radius_fuzzy must be >= spatial_radius_black")
        if self.temporal_radius_fuzzy < self.temporal_radius_black:
            raise ValueError("temporal_radius_fuzzy must be >= temporal_radius_black")
        if not 0.0 <= self.dropout_ratio <= 1.0:
            raise ValueError("dropout_ratio must be in [0, 1]")
        if self.dropout_radius < 0.0:
            raise ValueError("dropout_radius must be >= 0")
        if not 0.0 <= self.ema_mix_ratio <= 1.0:
            raise ValueError("ema_mix_ratio must be in [0, 1]")
        if self.ema_temperature <= 0.0:
            raise ValueError("ema_temperature must be > 0")
        if not 0.0 <= self.ema_floor_eps <= 1.0:
            raise ValueError("ema_floor_eps must be in [0, 1]")
        if self.n_spatial_coords <= 0:
            raise ValueError("n_spatial_coords must be > 0")
        if self.posenc_n_freqs <= 0:
            raise ValueError("posenc_n_freqs must be > 0")

        # 1. Input Processing
        self.patch_embed = PatchEmbed(fs, patch_seconds, overlap_seconds, embed_dim)

        # We calculate patch_size and step from the component we just initialized
        self.patch_size = self.patch_embed.patch_size
        self.step = self.patch_embed.step

        # 2. Positional Encoding (Shared between Encoder and Decoder)
        self.pos_enc = PosEnc(
            n_freqs=self.posenc_n_freqs,
            embed_dim=embed_dim,
            n_coords=self.n_spatial_coords + 1,
        )

        # 3. Encoder
        self.encoder = TransformerEncoderDecoder(embed_dim=embed_dim, depth=encoder_depth, heads=encoder_heads)

        # 4. Decoder (Main Reconstruction)
        self.decoder = MAEDecoder(embed_dim=embed_dim, decoder_depth=decoder_depth, decoder_heads=decoder_heads, patch_size=self.patch_size)

        self._init_megatron_transformer_weights()

        # 5. Auxiliary Head (Global Token)
        # We concatenate outputs from ALL encoder layers
        self.aux_dim = encoder_depth * embed_dim

        # A learned query vector to look at the encoder outputs
        self.aux_query = nn.Parameter(torch.randn(1, 1, self.aux_dim))

        # Projection: (Depth * Dim) -> Dim
        self.aux_linear = nn.Linear(self.aux_dim, embed_dim, bias=False)

        # Reconstruction Head for Aux Task
        self.aux_predict = nn.Sequential(nn.Linear(embed_dim, 2 * embed_dim), GEGLU(), nn.Linear(embed_dim, self.patch_size))

    def _init_megatron_transformer_weights(self):
        # Megatron-style init for transformer stack only; other params keep default init.
        for module in [self.encoder, self.decoder.decoder]:
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.normal_(submodule.weight, mean=0.0, std=0.02)
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)
                elif isinstance(submodule, nn.MultiheadAttention):
                    nn.init.normal_(submodule.in_proj_weight, mean=0.0, std=0.02)
                    if submodule.in_proj_bias is not None:
                        nn.init.zeros_(submodule.in_proj_bias)
                    nn.init.normal_(submodule.out_proj.weight, mean=0.0, std=0.02)
                    if submodule.out_proj.bias is not None:
                        nn.init.zeros_(submodule.out_proj.bias)

    def prepare_coords(self, xyz: torch.Tensor, num_patches: int):
        B, C, _ = xyz.shape
        if xyz.shape[-1] != self.n_spatial_coords:
            raise ValueError(
                f"Expected xyz last dim == {self.n_spatial_coords}, got {xyz.shape[-1]}"
            )
        device = xyz.device

        # 2. Generate Time Indices (0, 1, 2, ... P-1)
        time_idx = torch.arange(num_patches, device=device, dtype=torch.float32)

        # 3. Expand Spatial Coords
        # (B, C, S) -> (B, C, 1, S) -> (B, C, P, S)
        spat = xyz.unsqueeze(2).expand(-1, -1, num_patches, -1)

        # 4. Expand Time Coords
        # (P,) -> (1, 1, P, 1) -> (B, C, P, 1)
        time = time_idx.view(1, 1, num_patches, 1).expand(B, C, -1, -1)

        # 5. Concatenate -> (B, C, P, S+1)
        coords = torch.cat([spat, time], dim=-1)

        # 6. Flatten to (B, N_Total, S+1)
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

        # Keep original channels and append pairwise channels:
        # C -> C + C*(C-1)/2
        # Original xyz are repeated to match 6D pair coordinate schema.
        xyz_orig = torch.cat([xyz, xyz], dim=-1)
        x_all = torch.cat([x, x_pair], dim=1)
        xyz_all = torch.cat([xyz_orig, xyz_pair], dim=1)
        return x_all, xyz_all

    def _sample_mask_from_ema(
        self,
        ema_scores: torch.Tensor,
        batch_size: int,
        num_channels: int,
        num_patches: int,
        device: torch.device,
        base_mask: torch.Tensor | None = None,
        target_masked: int | None = None,
    ) -> torch.Tensor:
        N = num_channels * num_patches
        num_masked_target = int(self.mask_ratio * N) if target_masked is None else int(target_masked)
        num_masked_target = min(max(0, num_masked_target), N)

        if ema_scores.numel() != N:
            raise ValueError(
                f"EMA scores size mismatch: expected {N}, got {ema_scores.numel()}"
            )

        logits = ema_scores.reshape(-1).to(device=device, dtype=torch.float32)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        probs = torch.softmax(logits / self.ema_temperature, dim=0)
        if self.ema_floor_eps > 0.0:
            probs = (1.0 - self.ema_floor_eps) * probs + (self.ema_floor_eps / float(N))
        probs = probs / probs.sum().clamp_min(1e-12)

        if base_mask is None:
            mask = torch.ones((batch_size, N), dtype=torch.bool, device=device)
        else:
            if base_mask.shape != (batch_size, N):
                raise ValueError(
                    f"base_mask shape mismatch: expected {(batch_size, N)}, got {tuple(base_mask.shape)}"
                )
            mask = base_mask.clone()

        if num_masked_target > 0:
            for b in range(batch_size):
                curr_masked = int((~mask[b]).sum().item())
                need = num_masked_target - curr_masked
                if need <= 0:
                    continue

                visible_idx = torch.where(mask[b])[0]
                if visible_idx.numel() == 0:
                    continue

                if need >= int(visible_idx.numel()):
                    mask[b, visible_idx] = False
                    continue

                visible_probs = probs[visible_idx]
                visible_probs = visible_probs / visible_probs.sum().clamp_min(1e-12)
                pick_local = torch.multinomial(visible_probs, num_samples=need, replacement=False)
                pick_idx = visible_idx[pick_local]
                mask[b, pick_idx] = False

        return mask

    @staticmethod
    def _ratio_for_exact_count(num_to_mask: int, total: int) -> float:
        if total <= 0:
            return 0.0
        if num_to_mask <= 0:
            return 0.0
        if num_to_mask >= total:
            return 1.0
        return float(num_to_mask + 0.01) / float(total)

    def _sample_block_mask(
        self,
        coords: torch.Tensor,
        num_channels: int,
        num_patches: int,
        num_masked_target: int,
    ) -> torch.Tensor:
        N = num_channels * num_patches
        if num_masked_target <= 0:
            return torch.ones((coords.shape[0], N), dtype=torch.bool, device=coords.device)
        if num_masked_target >= N:
            return torch.zeros((coords.shape[0], N), dtype=torch.bool, device=coords.device)

        block_ratio = self._ratio_for_exact_count(num_masked_target, N)
        if self.which_mask == "default":
            return generate_mask(
                coords,
                mask_ratio=block_ratio,
                spatial_radius=self.spatial_radius_black,
                temporal_radius=self.temporal_radius_black,
                n_spatial_coords=self.n_spatial_coords,
            )

        block_mask, _ = generate_partial_mask(
            coords,
            num_channels=num_channels,
            num_patches=num_patches,
            mask_ratio=block_ratio,
            spatial_radius_black=self.spatial_radius_black,
            spatial_radius_fuzzy=self.spatial_radius_fuzzy,
            temporal_radius_black=self.temporal_radius_black,
            temporal_radius_fuzzy=self.temporal_radius_fuzzy,
            dropout_ratio=self.dropout_ratio,
            dropout_radius=self.dropout_radius,
            n_spatial_coords=self.n_spatial_coords,
        )
        return block_mask

    def _fuzzy_halo(self, mask: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        # mask: (B, N) with False at masked positions
        if self.which_mask != "fuzzy":
            return torch.zeros_like(mask)

        B, N, _ = coords.shape
        halo = torch.zeros((B, N), dtype=torch.bool, device=coords.device)

        for b in range(B):
            masked_idx = torch.where(~mask[b])[0]
            if masked_idx.numel() == 0:
                continue

            spatial_all = coords[b, :, : self.n_spatial_coords]
            spatial_masked = spatial_all[masked_idx]
            temporal_all = coords[b, :, self.n_spatial_coords]
            temporal_masked = coords[b, masked_idx, self.n_spatial_coords]

            d_spatial = torch.cdist(spatial_all, spatial_masked)
            d_temporal = (temporal_all.unsqueeze(1) - temporal_masked.unsqueeze(0)).abs()

            halo_b = (d_spatial <= self.spatial_radius_fuzzy) & (d_temporal <= self.temporal_radius_fuzzy)
            halo[b] = halo_b.any(dim=1) & mask[b]

        return halo

    def num_patches_for_length(self, total_samples: int) -> int:
        if total_samples < self.patch_size:
            return 0
        return int(((total_samples - self.patch_size) // self.step) + 1)

    def forward(
        self,
        x: torch.Tensor,
        xyz: torch.Tensor,
        mask_strategy: str = "random",
        ema_scores: torch.Tensor | None = None,
        return_token_l1: bool = False,
    ):
        x, xyz = self._to_pairwise_channels(x, xyz)
        B, _, _ = x.shape

        # --- 1. Patchify & Embed ---
        # patches: (B, C, P, PatchSize)
        patches = x.unfold(-1, self.patch_size, self.step)
        num_patches = patches.shape[2]

        # tokens: (B, C, P, Dim)
        tokens = self.patch_embed.linear(patches)

        # Flatten to Sequence: (B, N_Total, Dim)
        tokens_flat = tokens.flatten(1, 2)
        patches_flat = patches.flatten(1, 2)  # Target for loss

        # --- 2. Prepare Coordinates (S spatial + 1 temporal) ---
        coords = self.prepare_coords(xyz, num_patches)

        # --- 3. Generate Mask ---
        # Returns mask where counts are GUARANTEED to be equal across batch
        fuzzy_mask = None
        if mask_strategy == "ema" and ema_scores is not None:
            N = x.shape[1] * num_patches
            num_masked_target = int(self.mask_ratio * N)
            num_masked_target = min(max(0, num_masked_target), N)
            num_ema = int(round(self.ema_mix_ratio * num_masked_target))
            num_ema = min(max(0, num_ema), num_masked_target)
            num_block = num_masked_target - num_ema

            block_mask = self._sample_block_mask(
                coords=coords,
                num_channels=x.shape[1],
                num_patches=num_patches,
                num_masked_target=num_block,
            )
            mask = self._sample_mask_from_ema(
                ema_scores=ema_scores,
                batch_size=B,
                num_channels=x.shape[1],
                num_patches=num_patches,
                device=x.device,
                base_mask=block_mask,
                target_masked=num_masked_target,
            )
            fuzzy_mask = self._fuzzy_halo(mask, coords)
        elif self.which_mask == "default":
            mask = generate_mask(
                coords,
                mask_ratio=self.mask_ratio,
                spatial_radius=self.spatial_radius_black,
                temporal_radius=self.temporal_radius_black,
                n_spatial_coords=self.n_spatial_coords,
            )
        else:
            mask, fuzzy_mask = generate_partial_mask(
                coords,
                num_channels=x.shape[1],
                num_patches=num_patches,
                mask_ratio=self.mask_ratio,
                spatial_radius_black=self.spatial_radius_black,
                spatial_radius_fuzzy=self.spatial_radius_fuzzy,
                temporal_radius_black=self.temporal_radius_black,
                temporal_radius_fuzzy=self.temporal_radius_fuzzy,
                dropout_ratio=self.dropout_ratio,
                dropout_radius=self.dropout_radius,
                n_spatial_coords=self.n_spatial_coords,
            )

        # --- 4. Prepare Encoder Input ---
        # We need to extract only the visible tokens and stack them.
        # Since counts are fixed, we can do this efficiently using boolean masking and reshaping.

        # tokens_flat: (B, N_Total, D)
        # mask: (B, N_Total)
        # Result: (B, N_Vis, D)
        # The .view() works because the number of Trues in mask is identical for every row b.
        n_vis = int(mask[0].sum().item())

        if self.which_mask == "fuzzy" and self.fuzzy_noise_std > 0.0:
            if fuzzy_mask is None:
                fuzzy_mask = torch.zeros_like(mask)
            fuzzy_visible = fuzzy_mask & mask
            noise = torch.randn_like(tokens_flat) * self.fuzzy_noise_std
            tokens_flat = torch.where(fuzzy_visible.unsqueeze(-1), tokens_flat + noise, tokens_flat)

        x_vis = tokens_flat[mask].view(B, n_vis, -1)
        coords_vis = coords[mask].view(B, n_vis, -1)

        # Add PE
        pe_vis = self.pos_enc(coords_vis)
        x_vis = x_vis + pe_vis

        # --- 5. Encoder Forward ---
        x_encoded, intermediates = self.encoder(x_vis)

        # --- 6. Main Decoder Path ---
        predictions_main = self.decoder(x_visible=x_encoded, pos_enc=self.pos_enc, coords=coords, mask=mask)

        # --- 7. Auxiliary Path (Global Token) ---
        # Concatenate all intermediate layers: (B, N_Vis, Depth*Dim)
        aux_input = torch.cat(intermediates, dim=-1)

        # Attention Pooling
        # Score = Input @ Query.T
        # (B, N_Vis, AuxDim) @ (1, 1, AuxDim).T -> (B, N_Vis, 1)
        attn_scores = torch.matmul(aux_input, self.aux_query.transpose(1, 2))
        attn_weights = F.softmax(attn_scores, dim=1)

        # Pool: Sum(Weights * Input) -> (B, 1, AuxDim)
        global_token = torch.sum(attn_weights * aux_input, dim=1, keepdim=True)

        # Project to Embed Dim: (B, 1, Dim)
        global_emb = self.aux_linear(global_token)

        # Predict Masked Patches
        # 1. Get coords of masked tokens
        # Since mask is fixed count, we can reshape cleanly
        n_masked = (~mask[0]).sum().item()
        coords_masked = coords[~mask].view(B, n_masked, -1)

        pe_masked = self.pos_enc(coords_masked)

        # 2. Expand global token
        global_expanded = global_emb.expand(-1, n_masked, -1)

        # 3. Combine & Predict
        aux_pred_in = global_expanded + pe_masked
        predictions_aux = self.aux_predict(aux_pred_in)

        # --- 8. Loss Calculation ---
        # Target: Only the masked patches
        target_masked = patches_flat[~mask].view(B, n_masked, -1)

        # Main Loss (L1 on masked)
        pred_main_masked = predictions_main[~mask].view(B, n_masked, -1)
        loss_main = F.l1_loss(pred_main_masked, target_masked)

        # Aux Loss (L1 on masked)
        loss_aux = F.l1_loss(predictions_aux, target_masked)

        total_loss = loss_main + self.aux_loss_weight * loss_aux

        masked_l1 = (pred_main_masked - target_masked).abs().mean(dim=-1)
        token_l1 = torch.zeros((B, mask.shape[1]), device=predictions_main.device, dtype=masked_l1.dtype)
        token_l1[~mask] = masked_l1.view(-1)

        if return_token_l1:
            return total_loss, predictions_main, mask, token_l1
        return total_loss, predictions_main, mask
