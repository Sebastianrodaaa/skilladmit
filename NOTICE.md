# NOTICE

skilladmit's own code and text are MIT (see `LICENSE`). It reads, derives from, or ships the following third-party
material, each under its own terms.

## Data: public agent skills (`data/skills.jsonl.gz`)

`data/skills.jsonl.gz` holds, for 338 public skills, the `name` and `description` from each `SKILL.md` frontmatter
and the bullet lines of its "When to use" section, plus repo, commit, path and license. Only skills whose license is
MIT or Apache-2.0 (repo-level, or the skill's own `LICENSE*` file when present) are kept; every record names its
source file, so the copyright notice of the source repository applies to it. `make data` re-clones the pinned
revisions into `data/src/` (gitignored, not redistributed).

| repo | commit | license | skills kept |
|---|---|---|---|
| [wshobson/agents](https://github.com/wshobson/agents) | `7e2fb559cf4b824b27f1fec49fe88f7511066d26` | MIT, (c) 2024 Seth Hobson | 114 |
| [github/awesome-copilot](https://github.com/github/awesome-copilot) | `8178d1939a203f691067073dcab2a3f1e14072d3` | MIT, (c) GitHub, Inc. | 75 |
| [alirezarezvani/claude-skills](https://github.com/alirezarezvani/claude-skills) | `c85b17e1fb621ba760478baaacc4fcb9be225e74` | MIT, (c) 2025 Alireza Rezvani | 65 |
| [K-Dense-AI/claude-scientific-skills](https://github.com/K-Dense-AI/claude-scientific-skills) | `a25cf9ec00e2745d03f8704acf6f6919694ccd9a` | MIT, (c) 2025 K-Dense Inc. | 50 |
| [microsoft/skills](https://github.com/microsoft/skills) | `db273d02235f048331ca1597c0a98d529da1cabc` | MIT, (c) Microsoft Corporation | 23 |
| [huggingface/skills](https://github.com/huggingface/skills) | `25910b3d0ba31a2ba102f901cdc9e008c278a69c` | Apache-2.0 | 4 |
| [obra/superpowers](https://github.com/obra/superpowers) | `2ce8aa15d05bd57213e6f746f953aafb7ded7a8b` | MIT, (c) 2025 Jesse Vincent | 4 |
| [openai/skills](https://github.com/openai/skills) | `49f948faa9258a0c61caceaf225e179651397431` | per-skill `LICENSE.txt` (Apache-2.0 kept) | 3 |
| [anthropics/skills](https://github.com/anthropics/skills) | `ca45fc496a3436251f191354091d557da31c15b5` | per-skill `LICENSE.txt` | 0 (open-licensed skills have < 4 "When to use" bullets; docx/pdf/pptx/xlsx are not open-licensed) |

SYNTH candidates (`synth=True`, id prefix `SYNTH:`) are string edits of these records made at run time
(`skilladmit/data.py`, `synth_dup`, `synth_blend`); they are not published anywhere.

## Model (optional judge baseline only)

- **Qwen2.5-0.5B-Instruct** (Qwen team, Alibaba Cloud), revision `7ae557604adf67be50417f59c2c2f167def9a775`,
  Apache-2.0. Downloaded from Hugging Face at run time by `make judge`, not redistributed.

## Art and fonts (banner only, `scripts/banner.py`)

- Gustave Doré, *The Confusion of Tongues* (c. 1865-1868), wood engraving. Public domain.
  [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Confusion_of_Tongues.png).
- Albrecht Dürer, *Saint Jerome in His Study* (1514), engraving, National Gallery of Art. CC0 / public domain.
  [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Albrecht_D%C3%BCrer,_Saint_Jerome_in_His_Study,_1514,_NGA_6642.jpg).
- Pirata One by Rodrigo Fuenzalida and Nicolas Massi, SIL Open Font License 1.1
  ([google/fonts](https://github.com/google/fonts/tree/main/ofl/pirataone)). Rasterised into the banner; the font
  file is not redistributed.
- DejaVu Sans Mono, Bitstream Vera / DejaVu license. Rasterised into the banner.

## Software

NumPy and SciPy (BSD-3-Clause), scikit-learn (BSD-3-Clause), PyYAML (MIT), matplotlib (matplotlib license), Pillow
(MIT-CMU); optional: PyTorch (BSD-3-Clause), Hugging Face Transformers (Apache-2.0), Playwright (Apache-2.0, demo
capture only). The LTT and dashboard scaffolding follow the author's earlier
[prefixgate](https://github.com/Sebastianrodaaa/prefixgate) repo (MIT).
