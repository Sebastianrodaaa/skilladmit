## 2026-11-03 evening: skilladmit (LTT-gated admission of new agent skills via a typed decision model)
Status: proposed to Reverse Engineer Bot; needs Sebastian's go before any Devin session. Devin model: SWE-2 only. No Cursor cloud agents. No GPU spend; CPU / free compute only.

Signals combined: agent-skills (0.15 to 0.30, Skill++ lifelong routing interference), decision-policy-gates (0.40 to 0.45) and dllm-decision-engines (0.65 to 0.70, open Sarvam Prism).

Repo name: skilladmit
GitHub topics / hashtags: ai-frontier-research, agent-skills, decision-models, conformal-prediction, skill-routing, lifelong-learning, system-one, cpu-inference, typed-decisions

Idea: Skill++ shows that under progressive disclosure, adding a similar new skill can displace an earlier one even when the earlier skill and the router are unchanged. Policy Gates / Prism show typed decision models can act as fast allow/deny/route classifiers. skilladmit inserts a typed decision model as an ADMISSION GATE when a skill library grows: before a candidate skill is published into the live library, a small System One model over the fixed option set {admit, reject, rewrite_desc} reads the candidate name+description, a sample of existing skill metadata, and a held-out routing task set; Learn-then-Test (Angelopoulos et al. 2021) picks thresholds so that the post-admission misroute rate on previously correct tasks stays at most alpha (for example 0.05) with probability 1 - delta. Rejected skills stay in a staging index; rewrite_desc asks for a description edit and retests. The live router (name+description retrieval, as in Skill++) is unchanged; only what enters the library changes. v1 uses a hashing-vectorizer + multinomial logistic / small sklearn or tiny transformers classifier trained on synthetic-but-labeled routing tasks built from real public skill metadata (Anthropic/Claude skill dumps or SkillBench-style sets where available); optional arm swaps in a frozen open decision model head pattern inspired by sarvamai/prism if it runs on CPU within budget (otherwise document skip).

Why new (closest prior work, checked 2026-11-03 evening on arXiv API, HF papers search, GitHub search):
- Skill++ (https://arxiv.org/abs/2611.00544v1): diagnoses routing interference and proposes routing-aware skill refinement; it does not LTT-gate admission of new skills with a typed decision model.
- TalentRL / Structure-Aware Skill Retrieval (https://arxiv.org/abs/2610.41143v1, https://arxiv.org/abs/2610.41909v1): RL or retrieval for skill choice, not conformal library admission.
- Decision Models as Policy Gates (https://arxiv.org/abs/2611.02155v1): gates tool calls, not skill-library growth.
- prefixgate (our 2026-11-03 morning project): streaming LTT gate on tool-call prefixes, not skill admission.
- SEEK / Skill-RAG (https://arxiv.org/abs/2609.29803v1, https://arxiv.org/abs/2604.15771v4): skill routing for search/RAG, not lifelong admission control.
- GitHub: Intent-Router, laya-router-skill, zeph skills memory; none implement LTT-calibrated admit/reject on library growth. HF papers search for lifelong skill routing returned unrelated hits.
No repo matched "skill admission" + conformal / decision-model gate.

Hypothesis (falsifiable): on a held-out lifelong skill stream (skills added one by one; labels = gold skill for each routing query from a frozen task set), (1) skilladmit keeps post-admission misroute rate inside the LTT alpha bound (empirical rate inside the binomial 95% interval of alpha); (2) at matched final library size, end-of-stream routing accuracy is at least 5 points absolute above naive always-admit, and at least 3 points above a description-similarity reject baseline (reject if max cosine to existing descs > tau tuned on the same calib split). It fails if either bound is violated, if the CI on the gap vs the better baseline includes 0, or if always-admit already stays under alpha.

Experiment (CPU only, this box ~8 vCPU; measured microbenchmark: HashingVectorizer + LogisticRegression on 2000 docs / 50 classes ~2.2 s wall; full v1 estimate 20 to 45 min wall including data build, LTT grid, 3 seeds):
1. Build a lifelong stream: start with K0 public skill name+description pairs; add K_new candidates with controlled near-duplicates (real metadata + labeled paraphrase near-dupes; paraphrase model optional, or use string edits documented as synthetic and labeled SYNTH in data cards).
2. Router: progressive-disclosure baseline (embed name+desc, top-1); optional Skill++-style refinement arm if code is public, else cite as unavailable.
3. Admission arms: always-admit, similarity-threshold reject, prompted LLM judge (small open CPU model if free, else skip and document), skilladmit typed DM + LTT.
4. Report misroute vs library size, admission rate, bound violations, wall-clock, 3 seeds, all runs including negatives.
5. Optional: score the same admission decisions with an open decision-model-style head if sarvamai/prism (or a tiny distilled stand-in) fits in RAM on CPU; otherwise note skip. No GPU.

Key files (ponytail: fewest files): `skilladmit/data.py` (stream builder, SYNTH labels documented), `skilladmit/dm.py` (typed admit/reject/rewrite classifier), `skilladmit/ltt.py` (Learn-then-Test over threshold grid), `skilladmit/run.py` (arms, metrics, bootstrap CIs), `scripts/plots.py`, `scripts/demo_gif.py`, `app/server.py`, `app/index.html`, `results/` (real JSON only), `test_skilladmit.py` (one file).

How to verify: `python3 test_skilladmit.py` on CPU with a tiny stream of 30 real+SYNTH skills: scores are a valid simplex, LTT returns no configuration when alpha is below the empirical floor, always-admit on a near-duplicate stream shows measurable routing drop, and skilladmit refuses at least one near-duplicate that the similarity baseline also flags (or document disagreement). `make repro` reruns from cached features; README tables regenerate from results/*.json.

UI refs (Pinterest board https://www.pinterest.com/roda0050/ui/ ; local pack /workspace/ui-refs/pinterest/; STYLE.md: dark monochrome pro dashboards, dense panels, graph/embedding viz, one restrained accent):
- https://www.pinterest.com/pin/e1_AfqPShzpSM_IVitZ0gIRB2w56kSN6md0bBMdrSMamXXp_p1XQ1Yf
- https://www.pinterest.com/pin/e1_AfqPShyl4XMcd6lhL9tcRbVhvW-d-JyV6TTbNmSku0a4qebs867n
- https://www.pinterest.com/pin/e1_AfqPShwFGgsyP-hRNmJUfw-o5ZmAGxjYmn9nvHlh7ikDAhcWRc9J
- https://www.pinterest.com/pin/e1_AfqPShx7syn7L98XHpz-Og4MmRFFEIF1BSmxSlIgrHaRR-mfJEw6
- https://www.pinterest.com/pin/e1_AfqPShyzlhjdaKNYcuPytUs8zwU7byhDd9tAOEUFbJo7HU5JBLqn
Local images: /workspace/ui-refs/pinterest/pin01.jpg … pin05.jpg. Dashboard: library size vs misroute curve, admission timeline, embedding scatter of skill descs with admit/reject colors.

Ponytail principles (paste into Devin brief):
1. YAGNI ladder: skip speculative needs; reuse; stdlib; already-installed deps; one line; only then minimum code.
2. Fewest files; deletion over addition; boring over clever; no unrequested abstractions.
3. Bug fix = root cause (one shared guard beats patches everywhere).
4. No mock data in results; MOCK fixtures only under tests/mock_data/ and labeled.
Full skill: /home/box/agent-data/workflows/ponytail/SKILL.md

License: MIT (third-party model/data terms in NOTICE.md).

README format (mandatory paste from /workspace/readme-ref/README-FORMAT.md):
# README format for every project (Sebastian, 2026-11-02)
Model: https://github.com/ghgoodreau/athanor/blob/main/README.md (copy here: athanor-README.md, banner: athanor-banner.png)
PALETTE: black and white only. Pure black background, white/grey type and art. No sepia, no color accents anywhere (banner, screenshots, diagrams, badges).

