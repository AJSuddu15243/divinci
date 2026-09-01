# divinci

Train a model to reproduce CC0 paintings as compact p5.js / p5.brush programs.
One rollout is: painting → model → p5 program → headless render → similarity reward.

## Corpus

```sh
uv run scripts/fetch_paintings.py               # 1500 paintings
uv run scripts/fetch_paintings.py --count 50    # a small slice to look at
uv run scripts/fetch_paintings.py --area 400000 # bigger canvases
```

Writes `data/images/<source>-<id>.png` and `data/manifest.jsonl`. `data/` is
gitignored — re-running rebuilds it, and a bigger `--count` keeps what is already
downloaded. The script has no install step: dependencies are declared inline
(PEP 723) and `uv run` handles them.

Sources are the Cleveland Museum of Art and the Art Institute of Chicago, both
CC0 with keyless APIs. The Rijksmuseum was dropped — its old API returns 410 and
the replacement is a different integration.

## Two decisions worth knowing

**Equal-area canvases at the painting's own proportions.** Every canvas holds
200,000 pixels; the aspect ratio is whatever the painting is. Nothing is cropped,
padded, or upscaled.

Equal area means every rollout costs the same to render and reward scores compare
directly, with no per-pixel normalising. Free proportions mean the whole
composition survives — the earlier scheme snapped to seven fixed ratios and cut
away 4.6% of the picture on average.

Original pixel dimensions are not an option: the largest painting in the set is
47,217 x 14,073, which exceeds the ~16,384 max texture dimension WebGL will
allocate, and the median original is 117x the render cost of a 200k canvas.

**Medium filter.** Museums file manuscript folios, palm-leaf pages and
calligraphy scrolls under "Painting"; only about a third of what the catalogues
return is easel painting. The script keeps oil, acrylic and true tempera, which
is what brush strokes can plausibly reproduce. `--all-media` keeps everything.

## Rules for agents

These are not preferences. Do not restate them as trade-offs and do not work around them.

1. **Never write comments into a script.** No `#` comments, no docstrings, no
   exceptions. Code stays bare. If something needs explaining, it goes in
   `.agent/`, not in the file.
2. **Writing to `.agent/` is allowed** when there is something worth recording:
   `.agent/bin/agent-memory remember` for a verified fact,
   `.agent/bin/agent-memory decide` for a choice that constrains later work.
3. **Never write tests unless told to.** Not "while I was there", not as a
   sanity check.

Two lines in `scripts/fetch_paintings.py` look like comments but are not, and
deleting them breaks the script:

- `#!/usr/bin/env python3` — the shebang.
- The `# /// script ... # ///` block — PEP 723 metadata. `uv run` reads the
  dependency list from it. Remove it and the script cannot resolve Pillow.

## Conventions

Conventional commit prefixes: `feat` / `fix` / `test` / `docs` / `refactor` / `chore`.
Never commit credentials — the Tinker API key and anything like it lives in the
environment or an untracked `.env`.
