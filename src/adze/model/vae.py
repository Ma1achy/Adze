"""M2 — VAE mapping reasoning steps to latent blocks and back.

Encoder: a step's tokens -> K learned Perceiver query tokens cross-attending over
them -> [K, D]. Perceiver queries handle variable-length input natively, which
matters because a reasoning step is however many tokens the text is while a block
is exactly K latents.

Decoder: mirror. K latents -> token sequence for that step.

The decoder is non-autoregressive on purpose. An autoregressive decoder can reach
high reconstruction accuracy from teacher-forced tokens alone, ignoring the latent
entirely — which is exactly the failure the latent-use gate exists to catch, made
easy. Here the latent is the decoder's only input, so any accuracy above the
marginal has to come through it.

The hard gate for this milestone is the latent-use check, not reconstruction
accuracy. See adze.eval.checks.latent_use_check.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adze.data.tokeniser import MAX_STEP_LEN, PAD_ID


def _encoder_stack(d_model: int, n_layers: int, n_heads: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=d_model * 4,
        batch_first=True,
        norm_first=True,
        dropout=0.0,
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)


class StepEncoder(nn.Module):
    """Tokens for one step -> K latents.

    Small bidirectional transformer over the step's tokens, then K learned query
    tokens cross-attend over the result.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        latents_per_block: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, MAX_STEP_LEN, d_model) * 0.02)
        self.body = _encoder_stack(d_model, n_layers, n_heads)

        self.queries = nn.Parameter(torch.randn(1, latents_per_block, d_model) * 0.02)
        self.cross = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            tokens: [batch, T] token ids for one step.

        Returns:
            mu:     [batch, K, D]
            logvar: [batch, K, D]
        """
        batch, seq_len = tokens.shape
        pad_mask = tokens == PAD_ID

        h = self.embed(tokens) + self.pos[:, :seq_len]
        h = self.body(h, src_key_padding_mask=pad_mask)

        queries = self.queries.expand(batch, -1, -1)
        pooled, _ = self.cross(queries, h, h, key_padding_mask=pad_mask)
        pooled = self.norm(pooled)

        return self.to_mu(pooled), self.to_logvar(pooled)


class StepDecoder(nn.Module):
    """K latents -> tokens for one step.

    Mirror of the encoder: `max_len` learned position queries cross-attend over the
    K latents, then a bidirectional stack refines them into logits. All positions
    are produced in parallel — there is no token input, so the latent is the only
    thing the decoder has to work from.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        latents_per_block: int,
        latent_dim: int,
        max_len: int,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.from_latent = nn.Linear(latent_dim, d_model)
        self.latent_pos = nn.Parameter(torch.randn(1, latents_per_block, d_model) * 0.02)

        self.queries = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.cross = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

        self.body = _encoder_stack(d_model, n_layers, n_heads)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Args:
            latents: [batch, K, D]

        Returns:
            logits: [batch, T, vocab_size]
        """
        batch = latents.shape[0]
        memory = self.from_latent(latents) + self.latent_pos

        queries = self.queries.expand(batch, -1, -1)
        h, _ = self.cross(queries, memory, memory)
        h = self.norm(h + queries)
        h = self.body(h)

        return self.head(h)


class StepVAE(nn.Module):
    """Encoder + decoder + reparameterisation. Loss is CE reconstruction + beta * KL."""

    def __init__(self, encoder: StepEncoder, decoder: StepDecoder, kl_beta: float) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.kl_beta = kl_beta

        # Dedicated learned embedding for padding blocks — a trace has 3-7 steps
        # against a fixed B, and the leftover blocks need to be *something*
        # explicit rather than zeros or a sentinel value.
        #
        # It receives no gradient during M2: pad blocks are masked out of the VAE
        # loss by construction, so nothing here trains it. It exists now so the
        # cache and the denoiser have one canonical pad representation to share,
        # and it becomes a trained parameter in M3.
        latents_per_block = encoder.queries.shape[1]
        latent_dim = encoder.to_mu.out_features
        self.pad_latent = nn.Parameter(torch.randn(latents_per_block, latent_dim) * 0.02)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns dict with keys: loss, recon_loss, kl_loss, logits, latents."""
        mu, logvar = self.encoder(tokens)

        if self.training:
            std = torch.exp(0.5 * logvar)
            latents = mu + std * torch.randn_like(std)
        else:
            latents = mu

        logits = self.decoder(latents)

        recon_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tokens.reshape(-1),
        )
        # Summed over K and D, averaged over the batch.
        kl_loss = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=(1, 2)).mean()

        return {
            "loss": recon_loss + self.kl_beta * kl_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "logits": logits,
            "latents": latents,
        }


def build_vae(
    vocab_size: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    latents_per_block: int,
    latent_dim: int,
    kl_beta: float,
    max_len: int = MAX_STEP_LEN,
) -> StepVAE:
    """Assemble encoder + decoder + VAE from resolved config values."""
    encoder = StepEncoder(
        vocab_size, d_model, n_layers, n_heads, latents_per_block, latent_dim
    )
    decoder = StepDecoder(
        vocab_size, d_model, n_layers, n_heads, latents_per_block, latent_dim, max_len
    )
    return StepVAE(encoder, decoder, kl_beta)
