"""M2 — latent smoothness probe. Diagnostic, not a gate.

The latent-use check confirms the decoder *uses* the latent. With a
non-autoregressive decoder that is close to architecturally guaranteed — there is
no second information channel to collapse onto — so a large gap restates the
architecture rather than reporting on the model. It says nothing about whether the
latent is any *good*.

Reconstruction fidelity and representation effectiveness are close to unrelated.
A decoder can depend entirely on a latent space that is spiky and awkward, and the
symptom arrives at M3 disguised as "the denoiser won't train".

So: interpolate between two steps' latents and decode the midpoints. Smooth space,
plausible midpoints, and diffusion has something to work with. Spiky space, garbage
midpoints, and M3 fails for reasons that were decided here.

Nothing below cleans up the output. The decoded strings are raw argmax; the
well-formedness counts are measurements of what the model actually emits, not a
filter applied to it.

Usage:
    python scripts/m2_interpolate.py --checkpoint checkpoints/vae_debug.pt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from adze.config import load_config
from adze.data.generate import generate_dataset
from adze.data.tokeniser import MAX_STEP_LEN, CharTokeniser
from adze.model.vae import build_vae

STEP_RE = re.compile(r"^(-?\d+) ([+\-*]) (-?\d+) = (-?\d+)$")


def _classify(text: str) -> str:
    """well-formed and true / well-formed but false / malformed. Raw output only."""
    m = STEP_RE.match(text)
    if not m:
        return "malformed"
    lhs, op, rhs, result = int(m[1]), m[2], int(m[3]), int(m[4])
    expected = {"+": lhs + rhs, "-": lhs - rhs, "*": lhs * rhs}[op]
    return "true" if result == expected else "false"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/vae_debug.pt"))
    p.add_argument("--config", type=Path, default=Path("configs/debug.yaml"))
    p.add_argument("--pairs", type=int, default=500)
    p.add_argument("--show", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    config = load_config(args.config)
    device = torch.device(config.device)
    tokeniser = CharTokeniser()
    torch.manual_seed(args.seed)

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Prefer the architecture recorded in the checkpoint. A run trained with
    # --vae-layers does not match its config file, and rebuilding from the YAML
    # would construct the wrong shape.
    arch = state.get("arch") or {
        "vocab_size": tokeniser.vocab_size,
        "d_model": config.model.vae.d_model,
        "n_layers": config.model.vae.n_layers,
        "n_heads": config.model.vae.n_heads,
        "latents_per_block": config.model.latents_per_block,
        "latent_dim": config.model.latent_dim,
        "kl_beta": config.train.kl_beta,
        "max_len": MAX_STEP_LEN,
    }
    print(f"arch        {arch['n_layers']}L x {arch['d_model']}w, "
          f"K={arch['latents_per_block']} D={arch['latent_dim']}")
    vae = build_vae(**arch).to(device)
    vae.load_state_dict(state["model"])
    vae.eval()

    traces = generate_dataset(
        n=args.pairs * 2,
        seed=config.data.seed + 555_557,
        max_depth=config.data.max_depth,
        operand_max=config.data.operand_max,
    )
    texts = [s.render() for t in traces for s in t.steps][: args.pairs * 2]
    tokens = tokeniser.encode_batch(texts).to(device)

    with torch.no_grad():
        mu, _ = vae.encoder(tokens)

    left, right = mu[: args.pairs], mu[args.pairs : args.pairs * 2]
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    print(f"checkpoint  {args.checkpoint}")
    print(f"pairs       {args.pairs}\n")
    print("Decoded interpolations (raw argmax, first few pairs):\n")

    decoded: dict[float, list[str]] = {}
    with torch.no_grad():
        for alpha in alphas:
            z = (1 - alpha) * left + alpha * right
            preds = vae.decoder(z).argmax(dim=-1)
            decoded[alpha] = [tokeniser.decode(row) for row in preds]

    for i in range(min(args.show, args.pairs)):
        print(f"  {texts[i]!r}  ->  {texts[args.pairs + i]!r}")
        for alpha in alphas:
            text = decoded[alpha][i]
            print(f"      a={alpha:<5} {text!r:<24} {_classify(text)}")
        print()

    print("=" * 74)
    print("Well-formedness of decoded output by interpolation weight")
    print("=" * 74)
    print(f"  {'alpha':>6} {'well-formed':>13} {'of those, true':>16}")
    for alpha in alphas:
        kinds = [_classify(t) for t in decoded[alpha]]
        formed = sum(1 for k in kinds if k != "malformed")
        true = sum(1 for k in kinds if k == "true")
        share_true = true / formed if formed else 0.0
        print(f"  {alpha:>6} {formed / len(kinds):>12.1%} {share_true:>15.1%}")

    # Null baseline. Without it a well-formedness figure has no scale — and with
    # it, well-formedness turns out to measure nothing about the latent at all.
    with torch.no_grad():
        random_z = torch.randn_like(left)
        random_preds = vae.decoder(random_z).argmax(dim=-1)
    random_texts = [tokeniser.decode(row) for row in random_preds]
    random_kinds = [_classify(t) for t in random_texts]
    random_formed = sum(1 for k in random_kinds if k != "malformed")
    random_true = sum(1 for k in random_kinds if k == "true")
    print(f"  {'random':>6} {random_formed / len(random_kinds):>12.1%} "
          f"{random_true / max(random_formed, 1):>15.1%}   <- null baseline")

    print()
    print("  READ THE NULL ROW FIRST. If random latents are as well-formed as")
    print("  interpolated ones, well-formedness is the decoder's unconditional")
    print("  syntax prior — what it emits from anything — and says nothing about")
    print("  latent geometry. 'of those, true' is not a smoothness measure either:")
    print("  the midpoint of two true statements is almost never itself true.")

    # Round-trip consistency — the measure that actually discriminates.
    #
    # Decode a latent, re-encode what came out, and compare. A point ON the
    # manifold decodes to something whose encoding lands back where it started.
    # An off-manifold point decodes to text the encoder maps somewhere else
    # entirely. Unlike well-formedness this has a reason to separate interpolated
    # points from noise, because it asks whether the decoder and encoder agree
    # about the point rather than whether the output looks like arithmetic.
    print()
    print("=" * 74)
    print("Round-trip consistency: ||encode(decode(z)) - z|| / ||z||")
    print("=" * 74)
    norm_header = "|z'|/|z|"
    print(f"  {'alpha':>6} {'rel. error':>12} {'cosine':>9} "
          f"{norm_header:>10} {'re-encodable':>14}")

    def _round_trip(z: torch.Tensor) -> tuple[float, float, float, float]:
        with torch.no_grad():
            texts = [tokeniser.decode(row) for row in vae.decoder(z).argmax(dim=-1)]
        # A decode with no EOS can exceed max_len. Those are excluded from the
        # measurement rather than trimmed to fit — trimming would quietly improve
        # the number by editing the model's output.
        keep = [i for i, t in enumerate(texts) if len(t) + 2 <= MAX_STEP_LEN]
        if not keep:
            return float("nan"), float("nan"), float("nan"), 0.0
        toks = tokeniser.encode_batch([texts[i] for i in keep]).to(z.device)
        with torch.no_grad():
            mu2, _ = vae.encoder(toks)
        z_keep = z[keep]
        a, b = mu2.flatten(1), z_keep.flatten(1)
        rel = (a - b).norm(dim=1) / b.norm(dim=1)
        # Cosine separates "re-encodes somewhere wrong" from "re-encodes to the
        # origin": a relative error of exactly 1.0 is what mu2 -> 0 produces, and
        # that is the encoder pulling off-distribution text back to the prior mean
        # rather than disagreeing about where the point is.
        cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
        norm_ratio = a.norm(dim=1) / b.norm(dim=1)
        return (rel.mean().item(), cos.mean().item(),
                norm_ratio.mean().item(), len(keep) / len(texts))

    for alpha in alphas:
        z = (1 - alpha) * left + alpha * right
        rel, cos, nr, frac = _round_trip(z)
        print(f"  {alpha:>6} {rel:>12.3f} {cos:>9.3f} {nr:>10.3f} {frac:>13.1%}")
    rel, cos, nr, frac = _round_trip(random_z)
    print(f"  {'random':>6} {rel:>12.3f} {cos:>9.3f} {nr:>10.3f} {frac:>13.1%}"
          f"   <- null baseline")

    # Shell test. A norm ratio above 1 at the midpoint says re-encoding pushes the
    # point outward — which is what happens if latents live near a shell of roughly
    # constant radius and the straight-line midpoint falls inside it. If so the
    # midpoint is degraded by its *norm*, not by leaving the manifold, and
    # renormalising to the endpoints' radius should recover most of the gap.
    flat_l, flat_r = left.flatten(1), right.flatten(1)
    mid = 0.5 * (flat_l + flat_r)
    target_radius = 0.5 * (flat_l.norm(dim=1) + flat_r.norm(dim=1))
    scaled = mid * (target_radius / mid.norm(dim=1)).unsqueeze(1)
    rel, cos, nr, frac = _round_trip(scaled.view_as(left))
    print(f"  {'0.5 nrm':>6} {rel:>12.3f} {cos:>9.3f} {nr:>10.3f} {frac:>13.1%}"
          f"   <- midpoint rescaled to the endpoints' radius")

    print()
    print("  Low at alpha 0/1 and rising toward 0.5 is normal — the midpoint is the")
    print("  furthest from any real step. What matters is the gap to the random row.")
    print("  If midpoints round-trip no better than noise, the path between two real")
    print("  steps leaves the manifold, and M3 will read as 'the denoiser won't train'.")


if __name__ == "__main__":
    main()
