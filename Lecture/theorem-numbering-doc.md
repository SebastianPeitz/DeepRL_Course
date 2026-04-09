x # theorem-numbering.lua — Documentation

A Pandoc Lua filter for **automatic, continuous theorem/definition numbering**
across [decker](https://decker.cs.tu-dortmund.de) slide decks.

## What it does

- Numbers `definition`, `theorem`, `lemma`, `corollary`, `algorithm`, and
  `axiom` fenced divs automatically at **compile time**.
- Numbering is **continuous within each chapter** and spans multiple deck files
  (e.g. Definition 1.1–1.14 in `01-…`, then 1.15–1.19 in `02-…`).
- Supports optional **labels** and **cross-references** (same-file and
  cross-file), with **native Reveal.js slide navigation** — clicking a
  cross-reference jumps directly to the correct slide, no client-side
  JavaScript needed.
- Fully **order-independent** — each compilation scans *all* `*-deck.md` files
  from scratch, so `decker -S` watch mode works correctly regardless of which
  file is recompiled first.

## Files to share

| File | Purpose |
|------|---------|
| `theorem-numbering.lua` | The Lua filter (place next to your `*-deck.md` files) |
| `theorem-numbering-doc.md` | This documentation |

Everything else (decker.yaml changes, frontmatter cleanup) is described below.

## Setup

### 1. Place the filter

Copy `theorem-numbering.lua` into the same directory as your `*-deck.md` files
(i.e. the decker project root where `decker.yaml` lives).

### 2. Register in `decker.yaml`

Add the following to your `decker.yaml`:

```yaml
pandoc:
  filters:
    after:
      - theorem-numbering.lua
```

This tells decker to run the filter **after** its own Pandoc processing on every
deck file.  If you already have a `pandoc:` section, just merge the `filters`
key into it.

### 3. YAML frontmatter in each deck file

Each `*-deck.md` file needs a `chapter` key:

```yaml
---
subtitle: Learning from data
chapter: 1
---
```

Files sharing the same `chapter` value get **continuous numbering** across them,
ordered by filename (so `01-…` comes before `02-…`).

> **Note:** If you previously used an `offset` key for manual numbering, you can
> remove it — the filter computes offsets automatically.

### 4. Remove client-side numbering (if applicable)

If you have JavaScript that numbers theorems client-side (e.g. reading
`Decker.meta.offset`), remove or disable that code — the filter now handles
numbering at compile time.

## Usage

### Basic definition (no label)

```markdown
::: definition
A **tensor** is an element of $\mathbb{R}^{k_1 \times \ldots \times k_d}$.
:::
```

Output: **Definition 1.1** followed by the content.

### Labeled definition (referenceable)

```markdown
::: {.definition #def-tensor}
A **tensor** is an element of $\mathbb{R}^{k_1 \times \ldots \times k_d}$.
:::
```

Same output, but the label is **promoted to the slide's `<section>` element**
in the HTML output, so Reveal.js can navigate to it directly via URL hash.

> **Multiple definitions on one slide:** If a slide contains several labeled
> definitions, the first label becomes the `<section id="...">` and all
> others' cross-references automatically navigate to that same slide.

### Other environment types

```markdown
::: theorem
Every continuous function on a closed interval is bounded.
:::

::: lemma
If $f$ is differentiable, then $f$ is continuous.
:::

::: algorithm
1. Initialize $\theta$ randomly.
2. Repeat: $\theta \leftarrow \theta - \eta \nabla L(\theta)$.
:::
```

Output uses the configured display names (see *Customization* below):
**Satz 2.1**, **Lemma 2.2**, **Algorithmus 2.3**, etc.

### Inline mode

Adding `.inline` prepends the label to the first paragraph (with a trailing
dot and space) instead of inserting a separate paragraph:

```markdown
::: {.definition .inline #def-model}
A **model** is a parametrized family of functions.
:::
```

Output: **Definition 1.10.** A **model** is a parametrized family of functions.

### No-increment mode

Adding `.noinc` inserts a label with the *current* number but does not
increment the counter:

```markdown
::: {.definition .noinc}
(Continued from previous slide.)
:::
```

### Header inside a div

An `## H2` header inside the div is appended to the label and then removed:

```markdown
::: {.theorem #thm-bolzano}
## Bolzano-Weierstrass
Every bounded sequence has a convergent subsequence.
:::
```

Output: **Satz 3.1 Bolzano-Weierstrass** followed by the content (the H2 is
removed).

## Cross-references

### Same-file reference

```markdown
See [](#def-tensor) for the formal definition.
```

The empty link text `[]` is automatically filled in with the label, e.g.
"Definition 1.1".  The link target becomes `#/def-tensor` (Reveal.js hash
format), which navigates directly to the slide containing that definition.

### Cross-file reference

```markdown
See [](01-learning-from-data-deck.html#def-tensor).
```

Also auto-filled. Note: use the `.html` filename (the compiled output), not
`.md`.  The link becomes `01-learning-from-data-deck.html#/def-tensor` — when
clicked, Reveal.js opens the target deck and jumps to the correct slide.

> **Why `#/` instead of `#`?** Reveal.js uses `#/section-id` for slide
> navigation.  The filter adds the `/` automatically — you write plain
> `#label` in your markdown source.

### Unknown labels

If a label is not found, a warning is printed to stderr:

```
WARNING [theorem-numbering.lua]: unknown label 'def-foo'
```

## Customization

### Display names

The display names are defined near the top of `theorem-numbering.lua`:

```lua
local DISPLAY_NAMES = {
  definition = "Definition",
  theorem    = "Satz",
  lemma      = "Lemma",
  corollary  = "Korollar",
  algorithm  = "Algorithmus",
  axiom      = "Axiom",
}
```

Change these to match your language. For English:

```lua
local DISPLAY_NAMES = {
  definition = "Definition",
  theorem    = "Theorem",
  lemma      = "Lemma",
  corollary  = "Corollary",
  algorithm  = "Algorithm",
  axiom      = "Axiom",
}
```

### Adding new environment types

To add e.g. `example` or `remark`:

1. Add an entry to the `DISPLAY_NAMES` table in the Lua file.
2. Use `::: example` or `::: {.example #ex-foo}` in your markdown.
3. Add matching CSS for `.example` in your CSS file (background color, border,
   etc.).

### File ordering

Numbering order within a chapter is determined by **lexicographic sort of
filenames**. As long as your files are named with numeric prefixes
(`01-…`, `02-…`, `03-…`), the order is natural.

### Chapter grouping

Grouping is based on the `chapter` key in YAML frontmatter. Multiple files can
share the same chapter — numbering continues across them.

## How it works (internals)

1. **Directory scan**: On each invocation, the filter runs
   `ls <dir>/*-deck.md` to find all deck files.
2. **Raw text parsing**: For each file, reads the raw markdown, extracts
   `chapter` from YAML frontmatter, and counts `::: definition` (etc.) lines
   via pattern matching — skipping `.noinc` entries.
3. **Offset computation**: Files are sorted by name, grouped by chapter. Each
   file's starting number = 1 + sum of counts from earlier files in the same
   chapter.
4. **AST walk**: On the *current* file's Pandoc AST, the filter walks all
   `Div` elements with theorem classes, prepends a bold label, handles
   `.inline`/`.noinc`/headers, and sets div IDs.
5. **Reference resolution**: Walks all `Link` elements where the link text is
   empty and the target contains a known label — fills in the display text
   and rewrites the target to use `#/section-id` for Reveal.js navigation.
   If multiple labels share a slide, all references point to the first label's
   section ID.
6. **ID promotion**: Walks the top-level slide `Div` elements (decker wraps
   each slide in `Div {.slide .level1}`). For each slide, the first labeled
   theorem's ID is moved from the inner `<div>` to the slide `<section>`,
   so Reveal.js `#/label` navigation works natively.

No shared state, no registry files, no external tools. Each invocation is
self-contained.

## Testing without decker

You can test the filter directly with pandoc:

```bash
cd slides-and-notebooks/
pandoc --lua-filter theorem-numbering.lua -f markdown -t html 01-learning-from-data-deck.md \
  | grep '<strong>Definition'
```

## Requirements

- **Pandoc ≥ 2.17** (for Lua filter support with `pandoc.utils`)
- **Decker** with `pandoc.filters.after` support (any recent version)
- Each deck file must have `chapter: N` in its YAML frontmatter
- Deck filenames must match the `*-deck.md` pattern and sort lexicographically
  in the desired order
