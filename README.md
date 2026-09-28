# mtg-models

Trained checkpoints, ONNX exports, and per-epoch training history for the Magic: The Gathering
table-card detector used by [the-gathering](https://github.com/Nickjgniklu/the-gathering)'s
Super AI webcam-table feature. The code, synthetic data generator, real-capture golden dataset,
and full narrative for every result below live in that repo's `ml/` directory (see
`ml/README.md`, sections "Table scenes and the Super AI mode table detector" onward) — this repo
holds only the binary artifacts that don't belong in a source-code git history.

## Layout

- `runs/<name>/` — `best.pt` (the score-gated best checkpoint), `last.pt` (final epoch), `history.json`
  (per-epoch loss/recall/precision), `run.json` (exact CLI args, library versions, resume source).
- `exports/<version>/` — `table_detector.onnx`, `manifest.json`, `SHA256SUMS`: the standalone ONNX
  artifact `ml/cardid/export_table_detector.py` produces, verified against the torch reference.

## Training lineage

Each run warm-started from the previous one's `best.pt` (see each run's own `run.json` for the
exact resume path and CLI args):

1. **`table-a-pretrained-gpu`** (120 epochs, from an ImageNet-pretrained MobileNetV3-Small
   backbone) — the winning strategy after comparing three detection approaches; 95.8%/99.9%
   recall/precision on the original validation split.
2. **`table-a-nonstandard-finetune`** (32 epochs) — oversampled non-standard card frames
   (borderless, showcase, extended-art, etc.) after finding a real recall gap on them.
3. **`table-a-hardware-stress`** (21 epochs, stopped early at a healthy checkpoint after later
   epochs diverged) — dark-room, colour-cast, and lens-artifact augmentation after a real webcam
   capture showed the model struggling under bad lighting.
4. **`table-a-combined`** (12/30 epochs, interrupted by an OOM, never shipped) — attempted to
   combine a `distant_wide` camera-profile fix (a 0%-recall gap below 1/17 of frame width) with
   the hardware-stress augmentation; superseded by the next run before being confirmed better.
5. **`table-a-realcapture-hardening`** (20 epochs) — gaming-LED colour-cast bias, procedural
   clutter/round hard negatives, and deliberate card-stacking, after a real deployed-feature
   screenshot showed false positives on desk clutter and confused stacked cards.
6. **`table-a-hardneg-v4`** (20 epochs) — an explicit hard-negative loss upweight
   (`HARD_NEG_WEIGHT`) targeting the false-positive rate specifically; a real trade-off (cut
   synthetic hard-negative false positives but cost some broad recall), shipped anyway.
7. **`table-a-realclutter-v5`** (20 epochs) — **the current best/shipped checkpoint.** Trained
   against real photographed desk-clutter crops (dice, a deck box, a mouse, a keyboard, etc.)
   instead of only procedural shapes, closing most of the real-world gap a 108-card hand-verified
   golden dataset found: real-world precision 75.8% -> 87.4%, false-positive rate on known
   clutter 22.2% -> 18.5%, recall 92.6% -> 89.8%.
8. **`bgsub-a`** (10 epochs, parked experiment, separate `feat/background-subtraction-detector`
   branch) — a dual-input (background + current frame) architecture explored as an alternative
   way to suppress false positives on static clutter. Validated the core idea (false-positive
   rate on known negatives dropped to 4.5%) but recall regressed to 77.7% and did not recover
   over the full run; parked rather than continued.

`exports/2026-09-27-table-a-hardware-stress-ep17` is the ONNX export of `table-a-hardware-stress`'s
epoch-17 checkpoint (its last healthy epoch before the later divergence), from before the fully
history-tracked run above.

See `ml/README.md` in the main repo for the full investigation behind each step -- what broke,
how it was measured, and the same-seed before/after confusion-matrix numbers for every change.
