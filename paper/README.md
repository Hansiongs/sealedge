# sealedge JSS manuscript (`paper/`)

Plain LaTeX manuscript for Journal of Statistical Software (not `.Rnw`).

## Layout

| File | Role |
|------|------|
| `article.tex` | Manuscript |
| `refs.bib` | BibTeX |
| `jss.cls`, `jss.bst`, `jsslogo.jpg` | Official JSS style (minimal set) |
| `article.pdf` | Preview build (regenerable) |
| `README.md` | This file |

Build products (`*.aux`, `*.bbl`, `*.blg`, `*.log`, `*.out`) are gitignored.

## Build

Needs TeX Live / MiKTeX with `pdflatex` + `bibtex`.

```bash
cd paper
pdflatex article.tex
bibtex article
pdflatex article.tex
pdflatex article.tex
```

Or: `latexmk -pdf article.tex`

If compile dies at `\begin{document}`, check `\Plainauthor` / `\Plaintitle` / `\Plainkeywords` have no markup. Bare `_` in preamble meta must be `\_` (e.g. `\pkg{quant\_lib}`).

## Locked numbers

Case-study table must stay bit-identical to:

`../replication/output_paper_grade/results.json`

| Strategy | SPA p | OOS trades | ~runtime s |
|----------|-------|------------|------------|
| vol_compression_v1 | 1.0000 | 438 | 691 |
| pullback_sniper_rsi | 0.9300 | 180 | 345 |
| funding_rate_carry | 0.0005 | 3446 | 2138 |

Seed 42, `n_spa_iters=2000`, train_start 2021-07-01.

## Out of scope here

OJS upload, real PyPI 1.0 + Zenodo (post-manuscript, pre-submit).
