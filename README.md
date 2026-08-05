# Adze

A latent diffusion reasoner. Reasoning happens in continuous latent space; text
exists only at the interfaces.

A normal LLM writes like a typewriter — one token at a time, left to right, every
keystroke permanent. This drafts reasoning steps as continuous vectors, then
revises them globally, and only converts to text once the thinking is done.

- `docs/design.md` — what and why. The reference.
- `docs/build-plan.md` — the milestone sequence. M0 through M7.
- `CLAUDE.md` — agent instructions, invariants, and the not-in-v0 list.

## Status

v0 design frozen. Implementation not started.

## The first result

> Given the corrupted block's location, does global regeneration repair it more
> reliably than causal regeneration?

Everything else is downstream of that.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest              # most tests fail until their milestone lands
```

## Layout

```
src/adze/
  data/     synthetic arithmetic traces, corruption, dataset
  model/    VAE, denoiser, attention masks
  train/    training loops
  sample/   draft (pass one), refine (pass two), trajectory printer
  eval/     gates and the central experiment
configs/    debug.yaml (fast iteration), v0.yaml (real runs)
scripts/    one-off utilities
tests/      acceptance criteria, one module per milestone
```
