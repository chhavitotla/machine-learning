# Contributing to ml-zero-to-hero

Thanks for being here. This repo improves one concept at a time, and the bar is "would this help someone the night before an interview?" If yes, we want it.

## The three-file convention

Every concept folder contains exactly:

```
concept_name/
├── README.md           # intuition → math → usage → failure modes → 5 interview Q&As
├── visualize.html      # the algorithm running, interactively, zero dependencies
└── implementation.py   # from-scratch, stdlib-only Python with a __main__ demo
```

Use `01_supervised/linear_regression/` as the reference for all three.

## Rules for `visualize.html`

These are hard requirements — PRs that miss them will get a friendly nudge, not a merge:

1. **One file, zero dependencies.** No CDN scripts, no frameworks, no build step. Double-click → works, offline.
2. **Interactive.** At minimum one slider that changes a parameter live, or a step-through button showing the algorithm one iteration at a time. Both is better.
3. **Show the algorithm running**, not just the result. The user should watch it happen.
4. **Dark theme, clean typography.** Background `#0f0f13`, panels `#16161d`, accent colors from the existing files. It should look like distill.pub, not a homework assignment.
5. **Live annotations.** A panel that explains, in plain English, what the algorithm is doing *right now* — updating as it runs.
6. **Comment the algorithm, not the canvas code.** Explain why the math is what it is; nobody needs a comment on `ctx.beginPath()`.
7. **Real math only.** The numbers on screen must come from actually running the algorithm. No faked convergence, no hardcoded "results."

## Rules for `implementation.py`

1. **Standard library only.** No numpy, no sklearn. The point is that nothing hides inside a library call.
2. **Readable over fast.** Lists of floats, clear names, the README's equations mirrored in the comments with the same symbols.
3. **A `__main__` demo that proves it learns** — trains on a small problem and prints evidence (loss going down, correct predictions). It must run in seconds: `python3 implementation.py`.

## Rules for `README.md`

Follow the exact section order of the template: **The Intuition** (one paragraph, zero jargon) → **The Math** (every symbol defined, every equation followed by a plain-English sentence) → **Open the visualization** → **When to use this** (3 bullets) → **What breaks it** (3 bullets) → **5 Interview Questions** (one each: conceptual, mathematical, practical, gotcha, system design; every answer as *direct answer → reason → likely follow-up and how to handle it*).

## Other welcome contributions

- **GIFs** of the visualizations for the root README: 10–15 seconds, under 3 MB, into `docs/gifs/` with the filename already referenced in the README.
- **Better interview answers**, especially ones tested in real interviews.
- **Math/code corrections.** Correctness beats everything; a one-line fix PR is a great PR.
- **Typos and clarity edits.** Yes, really.

## Process

1. **Open an issue first for new topics** so two people don't build the same thing.
2. Fork → branch (`add-svm-visualization`) → PR. **One topic per PR.**
3. In the PR description, include a screenshot of your visualization and paste the output of your `implementation.py` demo.
4. No CLA, no bureaucracy. If it's correct, clear, and follows the conventions, it merges.

## The quality test

Before submitting, open your work and ask:

- Does the README read like a brilliant friend explaining over coffee, or a textbook?
- Does the visualization look like distill.pub, or a student project?
- Could someone who opens *only your folder* walk into an interview and handle this topic?

If any answer is wrong, it's not done yet. When they're all right, send it — and thank you.
