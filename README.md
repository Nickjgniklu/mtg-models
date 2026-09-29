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

## Reproduction lineage and tiled fusion (`feat/reproduce-table-detector-training`)

A from-scratch reproduction of the lineage above (see `ml/cardid/reproduce_table_detector.py`),
run to confirm the whole pipeline is actually rebuildable, not just documented:

1. **`repro-a-pretrained`** (120 epochs) -> **`repro-a-nonstandard-finetune`** (50 epochs, later
   found redundant -- kept in this archive for completeness, not because it did anything: the
   oversampled non-standard-card catalog it existed to introduce was already baked into the
   reused dataset from the first epoch) -> **`repro-a-hardware-stress`** (25 epochs) ->
   **`repro-a-realcapture-hardening`** (20 epochs) -> **`repro-a-hardneg-v4`** (20 epochs) ->
   **`repro-a-realclutter-v5`** (20 epochs, also found redundant for the same reason as
   `nonstandard-finetune` -- kept for completeness).
2. **`repro-a-hardneg-v4` is the best checkpoint of this reproduction**, not the last one: scored
   against the real-capture golden set directly (not synthetic `val`), it beats the final
   `repro-a-realclutter-v5` on every measure (recall 87.0%/precision 84.7%/fp_on_negatives 7.4% vs.
   84.3%/75.8%/18.5%) and is close to or better than the originally-shipped `table-a-realclutter-v5`
   above, especially on false positives. **Use `repro-a-hardneg-v4`, not `repro-a-realclutter-v5`,
   as this reproduction's output.**
3. **`repro-a-hardneg-v4-seed1`** -- a same-recipe retry from a different seed, to see if
   stochastic variance alone could close the small remaining gap to the historical checkpoint.
   It didn't (recall 88.0%/precision 83.3%/fp_on_negatives 11.1%, not better overall); kept for
   reference, not recommended for use.
4. **`tiled-fusion-a`** and **`tiled-fusion-1920`** -- `TableCenterNet` fixed, frozen (from
   `repro-a-hardneg-v4`); these `.pt` files are the trainable `FusionHead` *only* (~54k params),
   not a full detector checkpoint -- load them into `tiled_fusion.TiledFusionDetector.fusion`
   alongside `repro-a-hardneg-v4/best.pt` as the frozen base, not standalone. `tiled-fusion-a` is
   trained at 640x640 (the original table-scenes resolution); `tiled-fusion-1920` at 1920x1920
   (matching the deployed webcam's real capture resolution). **`tiled-fusion-1920` needs
   `--score-threshold 0.15-0.16`, not the project's usual 0.3** -- its bigger canonical grid
   produces confidence scores on a different scale; at the wrong threshold it looks like a
   recall/precision tradeoff against every other approach, but at the right one it beats single-pass,
   the heuristic tiled-inference dedupe, and `tiled-fusion-a` on both recall and precision at once
   (89.8%/89.8% vs. the previous best of 87.0%/87.9%). See `ml/README.md`'s "Learned tile fusion"
   section for the full numbers and the why.
5. **Both single-pass (`repro-a-hardneg-v4`) and tiled-fusion (`tiled-fusion-1920`) are kept
   long-term, not one replacing the other** -- single-pass for a smaller/focused capture area
   (fast: ~11ms CPU / ~5ms GPU per frame), tiled-fusion for a full-table scan at the fixed
   resolution the product always downscales incoming video to (~15x the CPU cost, ~1.6x on GPU).

`ml/cardid/detect_and_embed.py` (same branch) wires any of the detectors above to the existing
`Embedder` through one batched crop layer, so one forward pass produces every detected card's
identifying embedding directly -- no new checkpoint of its own, just a new way of composing
already-trained ones. Both the plain and tiled combinations export to ONNX cleanly (verified
against the torch model to floating-point noise).

`exports/repro-a-hardneg-v4/table_detector.onnx` is this lineage's recommended export -- use this,
not an export of `repro-a-realclutter-v5`, per point 2 above.

`exports/detect-and-embed-repro-a-hardneg-v4-placeholder-embed/detect_and_embed.onnx` -- an
earlier export with a random-weight placeholder embedder, kept for reference only. **Superseded
by the export below**, which has a real embedder.

**`runs/recogniser-cfbender-oracle/best.pt`** -- a real, currently-deployed `Embedder` checkpoint,
recovered from `cfbender/oracle`'s public `models/recogniser.pt` (see that directory's
`PROVENANCE.md`): sha256 `52b3528e...a0bf`, confirmed byte-identical to the `recogniser` entry in
the currently-deployed bundle's own `manifest.json`. Loads into `cardid.model.Embedder` with
`strict=True` (all 242 keys match). Spot-checked against the existing reference crop pipeline
(`detect.warp_card` + `detect.art_crop("modern")`) on 5 real detections: cosine similarity 0.997+
between the two crop methods' embeddings through this same checkpoint, confirming the crop math
and the recovered weights are both correct, not just structurally valid.

**`exports/detect-and-embed-repro-a-hardneg-v4-real-embed/detect_and_embed.onnx`** -- single-pass
detector + real embedder, verified against the torch model to floating-point noise. Tested
end-to-end (`ml/cardid/evaluate_detect_and_embed.py`): 98.4% detection recall, 72.7% modern-frame
top-1 identification against the real gallery. That identification number is capped by
`native_size=384`'s already-downsampled crop source, not a bug -- see the export below and
`ml/detect-and-embed-guide.md`'s "Tested end-to-end" section for the full explanation.

**`exports/detect-and-embed-tiled-fusion-1920-real-embed/detect_and_embed.onnx`** -- tiled-fusion
detector (native 1920 resolution) + real embedder, same verification. Tested the same way: 93.7%
detection recall, **93.0%** modern-frame top-1 identification -- matching the reference
`embed.onnx` pipeline's number, because cropping from the true native resolution (not an
already-downsampled 384px canvas) preserves the detail the embedder needs. **This is the
recommended export when identification quality matters**; the single-pass export above is
recommended when latency on CPU/WASM matters more (~23ms vs ~191ms per frame on this project's
desktop CPU -- the gap nearly disappears on GPU, ~20ms vs ~21ms, since embedding the cards found,
not detecting them, dominates GPU time for both). Needs `--score-threshold 0.15-0.16` at
inference time, not the project's usual 0.3 -- see `tiled_fusion.py` for why.

Both exports share the same known limitation, architectural, not a weights problem: this module
crops only the "modern" frame window per card (see `detect_and_embed.py`'s module docstring), so
its single embedding per card is not yet compatible with the real `search.onnx` (which expects 14
frame-hypothesis embeddings and looks up each gallery art's own frame index into that batch) --
see `ml/detect-and-embed-guide.md`'s "frame-hypothesis gotcha" section for the three documented
options to close that gap, none implemented yet. Non-modern-frame cards identify meaningfully
worse in both exports (~78-79% top-1) for exactly this reason.
